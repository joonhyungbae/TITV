#!/usr/bin/env python3
"""
00_build_dataset.py
===================
Reproducible pipeline: artist_details.json -> data.json

Transforms raw crawled artist data into the structured, enriched dataset
used by all downstream analysis scripts.

IMPORTANT NOTE ON REPRODUCIBILITY:
    The canonical data.json shipped with this replication package was
    originally enriched using LLM-assisted parsing (for Korean->English
    translation, institution classification, and event type detection),
    then cleaned for consistency (see steps 1-3 in the plan).

    This script provides a DETERMINISTIC, regex-based reconstruction
    that approximates the same transformation using pre-built mapping
    tables extracted from the canonical data.json. Minor differences
    in event counts may occur due to the regex vs LLM parsing gap.

    The canonical data.json should be used for analysis; this script
    documents the pipeline for transparency.

Input:
    data/artist_details.json         (raw crawled career text)
    data/institution_mappings.json   (canonical institution name/type/country)
    data/translation_mappings.json   (Korean->English translation tables)

Output:
    data/data.json                   (enriched, analysis-ready dataset)

Processing steps:
    1. Parse raw Korean career text (raw_text) into structured events
       by detecting section headers and extracting year + institution
    2. Translate Korean text to English using canonical mapping tables
    3. Classify institutions and events deterministically
    4. Compute summary/aggregate fields
    5. Ensure consistency (one English name and type per Korean institution)
"""

import json
import re
import os
import sys
from collections import Counter, defaultdict

# ============================================================
# Constants
# ============================================================

OVERSEAS_CITY_MARKERS = {
    '뉴욕', '도쿄', '파리', '런던', '베를린', '베이징', '상하이',
    '로스앤젤레스', 'LA', '시카고', '워싱턴', '보스턴',
    '오사카', '교토', '나고야', '후쿠오카',
    '밀라노', '로마', '베니스', '베네치아', '피렌체',
    '암스테르담', '브뤼셀', '취리히', '빈', '비엔나',
    '바르셀로나', '마드리드', '리스본',
    '모스크바', '상트페테르부르크',
    '시드니', '멜버른', '오클랜드',
    '상파울루', '멕시코시티', '카이로',
    '싱가포르', '홍콩', '타이베이', '마닐라', '자카르타',
    '방콕', '쿠알라룸푸르', '하노이',
    '뮌헨', '함부르크', '프랑크푸르트', '쾰른', '뒤셀도르프',
    '카셀', '드레스덴',
    '뉴헤이븐', '필라델피아', '샌프란시스코',
    '로잔', '제네바', '바젤',
}

# Country markers in text
COUNTRY_KO_TO_CODE = {
    '미국': 'US', '일본': 'JP', '프랑스': 'FR', '독일': 'DE',
    '영국': 'GB', '이탈리아': 'IT', '중국': 'CN', '스페인': 'ES',
    '네덜란드': 'NL', '벨기에': 'BE', '스위스': 'CH', '오스트리아': 'AT',
    '호주': 'AU', '캐나다': 'CA', '브라질': 'BR', '멕시코': 'MX',
    '러시아': 'RU', '인도': 'IN', '싱가포르': 'SG', '홍콩': 'HK',
    '대만': 'TW', '태국': 'TH', '필리핀': 'PH', '베트남': 'VN',
    '인도네시아': 'ID', '말레이시아': 'MY', '터키': 'TR',
    '폴란드': 'PL', '체코': 'CZ', '헝가리': 'HU', '그리스': 'GR',
    '덴마크': 'DK', '스웨덴': 'SE', '노르웨이': 'NO', '핀란드': 'FI',
    '이스라엘': 'IL', '이집트': 'EG', '남아공': 'ZA',
    '아르헨티나': 'AR', '콜롬비아': 'CO', '칠레': 'CL',
    '뉴질랜드': 'NZ', '아이슬란드': 'IS', '포르투갈': 'PT',
    '아일랜드': 'IE', '룩셈부르크': 'LU', '슬로바키아': 'SK',
    '크로아티아': 'HR',
}

CITY_TO_COUNTRY = {
    '뉴욕': 'US', '로스앤젤레스': 'US', 'LA': 'US',
    '시카고': 'US', '워싱턴': 'US', '보스턴': 'US',
    '샌프란시스코': 'US', '뉴헤이븐': 'US', '필라델피아': 'US',
    '도쿄': 'JP', '오사카': 'JP', '교토': 'JP', '나고야': 'JP', '후쿠오카': 'JP',
    '파리': 'FR', '런던': 'GB', '베를린': 'DE', '뮌헨': 'DE',
    '함부르크': 'DE', '프랑크푸르트': 'DE', '쾰른': 'DE',
    '뒤셀도르프': 'DE', '카셀': 'DE', '드레스덴': 'DE',
    '밀라노': 'IT', '로마': 'IT', '베니스': 'IT', '베네치아': 'IT', '피렌체': 'IT',
    '베이징': 'CN', '상하이': 'CN', '암스테르담': 'NL',
    '브뤼셀': 'BE', '취리히': 'CH', '로잔': 'CH', '제네바': 'CH', '바젤': 'CH',
    '빈': 'AT', '비엔나': 'AT',
    '바르셀로나': 'ES', '마드리드': 'ES',
    '모스크바': 'RU', '상트페테르부르크': 'RU',
    '시드니': 'AU', '멜버른': 'AU',
    '싱가포르': 'SG', '홍콩': 'HK', '타이베이': 'TW',
    '상파울루': 'BR', '멕시코시티': 'MX',
    '리스본': 'PT', '마닐라': 'PH', '방콕': 'TH',
    '오클랜드': 'NZ',
}

# Degree keywords
DEGREE_KEYWORDS = {
    'phd': ['박사'],
    'master': ['대학원 졸업', '대학원 수료', '석사', '대학원졸업', 'M.F.A', 'MFA',
               '동 대학원 졸업', '동대학원 졸업', '동 대학원졸업'],
    'bachelor': ['대학 졸업', '대학교 졸업', '대학졸업', '대학교졸업', 'B.F.A', 'BFA',
                 '학사'],
    'diploma': ['수료', '수학'],
}

# National exhibition (Gukjeon) patterns
NATIONAL_EXHIBITION_RE = re.compile(
    r'(국전|대한민국미술전람회|대한민국미술대전|대한민국 미술전람회|대한민국 미술대전)'
)

# Section header patterns in raw_text
SECTION_MAP = {
    # Solo exhibitions
    '개인전': 'solo_exhibition',
    '개 인 전': 'solo_exhibition',
    # Group exhibitions
    '단체전': 'group_exhibition',
    '그룹전': 'group_exhibition',
    '단체전 및 초대전': 'group_exhibition',
    # Awards
    '수상': 'award',
    '수 상': 'award',
    '수상경력': 'award',
    '3.수상경력': 'award',
    # Education
    '학력': 'education',
    '수련과정': 'education',
    '1.수련과정': 'education',
    # Career/positions
    '경력': 'position',
    '현재': 'position',
    '현직': 'position',
    '활동경력': 'position',
    '2.활동경력': 'position',
    '1)일반경력': 'position',
    # Collections
    '작품소장': 'collection',
    '작품 소장': 'collection',
}

# Keywords for classifying events within sections
HONOR_KEYWORDS = ['훈장', '표창', '포장', '공로상', '공로장', '감사장', '감사패']
RESIDENCY_KEYWORDS = ['레지던시', '레지던스', '연수', '초빙', 'residency', 'Residency']
BIENNALE_KEYWORDS = ['비엔날레', '비엔나레', 'biennale', 'Biennale', '트리엔날레']


# ============================================================
# Loading helpers
# ============================================================

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# Text parsing helpers
# ============================================================

def extract_year(text):
    """Extract a 4-digit year from text."""
    m = re.search(r'(\d{4})', text)
    return int(m.group(1)) if m else None


def extract_institution_and_city(text):
    """
    Extract institution name and optional city from exhibition entry text.
    Patterns:
      "YYYY년 제목 (기관, 도시)"
      "YYYY년 기관"
      "YYYY 기관, 도시"
    """
    # Remove year prefix
    cleaned = re.sub(r'^\d{4}년?\s*', '', text).strip()

    # Try to extract institution from parentheses
    paren_match = re.search(r'\(([^)]+)\)', cleaned)
    if paren_match:
        inner = paren_match.group(1)
        parts = [p.strip() for p in inner.split(',')]
        institution = parts[0]
        city = parts[1] if len(parts) > 1 else None
        return institution, city

    # No parentheses: the whole text after year is the institution
    parts = [p.strip() for p in cleaned.split(',')]
    if len(parts) >= 2:
        institution = parts[0]
        city = parts[-1]
    else:
        institution = cleaned
        city = None

    return institution, city


def extract_solo_institution(text):
    """
    Extract institution from solo exhibition entry.
    Pattern: "YYYY년 기관명" or "YYYY 기관명" or "YYYY년 전시명 (기관, 도시)"
    """
    cleaned = re.sub(r'^\d{4}년?\s*', '', text).strip()

    # If there are parentheses, extract from them
    paren_match = re.search(r'\(([^)]+)\)', cleaned)
    if paren_match:
        inner = paren_match.group(1)
        parts = [p.strip() for p in inner.split(',')]
        institution = parts[0]
        city = parts[1] if len(parts) > 1 else None
        event_name = cleaned[:cleaned.index('(')].strip()
        return event_name, institution, city

    # No parentheses: the whole text is the institution
    parts = [p.strip() for p in cleaned.split(',')]
    if len(parts) >= 2:
        institution = parts[0]
        city = parts[-1]
    else:
        institution = cleaned
        city = None

    return None, institution, city


def detect_country(text, city=None):
    """Detect country from text and city name."""
    # Check city first
    if city:
        city_clean = city.strip()
        if city_clean in CITY_TO_COUNTRY:
            return CITY_TO_COUNTRY[city_clean]
        for country_ko, code in COUNTRY_KO_TO_CODE.items():
            if country_ko in city_clean:
                return code
    # Check full text
    for city_name in OVERSEAS_CITY_MARKERS:
        if city_name in text:
            if city_name in CITY_TO_COUNTRY:
                return CITY_TO_COUNTRY[city_name]
    for country_ko, code in COUNTRY_KO_TO_CODE.items():
        if country_ko in text:
            return code
    return 'KR'


def detect_degree(text):
    """Detect degree level from education text."""
    for degree, keywords in DEGREE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return degree
    return 'other'


def is_overseas_institution(inst_ko, country, inst_type_map):
    """Check if an institution is overseas."""
    if country and country != 'KR':
        return True
    if inst_ko in inst_type_map:
        return inst_type_map[inst_ko] in ('overseas', 'overseas_university')
    return False


# ============================================================
# Raw text section parsing
# ============================================================

def detect_section(line):
    """
    Detect if a line is a section header.
    Returns the section type or None.
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Direct match
    if stripped in SECTION_MAP:
        return SECTION_MAP[stripped]

    # Fuzzy match: strip numbers and punctuation
    cleaned = re.sub(r'^[\d.)]+\s*', '', stripped)
    if cleaned in SECTION_MAP:
        return SECTION_MAP[cleaned]

    return None


def split_raw_text_into_sections(raw_text):
    """
    Split raw_text into sections based on section headers.
    Returns dict: section_type -> list of entry lines.
    """
    sections = defaultdict(list)
    current_section = None

    for line in raw_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line is a section header
        section = detect_section(stripped)
        if section is not None:
            current_section = section
            continue

        # If no section yet, try to detect birth_info or education from content
        if current_section is None:
            if re.match(r'^\d{4}\s*.*(출생|生)', stripped):
                sections['birth_info'].append(stripped)
                continue
            # Lines before first section with university names -> education
            if any(kw in stripped for kw in ['대학', '학교', 'University', 'College']):
                sections['education'].append(stripped)
                continue
            # Skip unclassifiable lines before first section
            continue

        sections[current_section].append(stripped)

    return sections


def parse_exhibition_entry(entry, event_type, inst_en_map, inst_type_map, inst_country_map):
    """
    Parse a single exhibition entry line into a structured event.
    Handles both solo and group exhibition formats.
    """
    year = extract_year(entry)
    if year is None:
        return None

    # Remove year prefix (handles YYYY, YYYY.MM formats)
    cleaned = re.sub(r'^\d{4}[년.]?\d{0,2}\s*', '', entry).strip()

    # Check for biennale keywords
    actual_type = event_type
    if any(kw in entry for kw in BIENNALE_KEYWORDS):
        actual_type = 'biennale'

    # Extract institution from parentheses
    institution, city = extract_institution_and_city(entry)

    # Determine country
    country = detect_country(entry, city)
    if institution and institution in inst_country_map:
        country = inst_country_map[institution]

    # Canonical English name and type
    inst_en = inst_en_map.get(institution, institution) if institution else None
    inst_type = inst_type_map.get(institution, 'other') if institution else 'other'

    # Event title
    paren_idx = cleaned.find('(')
    event_title = cleaned[:paren_idx].strip() if paren_idx > 0 else cleaned
    if event_type == 'solo_exhibition' and event_title:
        event_ko = f"개인전: {event_title}" if event_title != institution else f"개인전: {institution}"
    else:
        event_ko = event_title

    return {
        'year': year,
        'event_type': actual_type,
        'event_ko': event_ko,
        'event_en': None,
        'detail_ko': None,
        'detail_en': None,
        'institution_ko': institution,
        'institution_en': inst_en,
        'institution_type': inst_type,
        'country': country,
        'city_ko': city,
        'city_en': None,
    }


def parse_award_entry(entry, inst_en_map, inst_type_map, award_name_map, award_body_map):
    """Parse a single award entry line."""
    year = extract_year(entry)
    if year is None:
        return None, None

    cleaned = re.sub(r'^\d{4}[년.,]?\d{0,2}\s*', '', entry).strip()

    # Detect honor vs award
    is_honor = any(kw in entry for kw in HONOR_KEYWORDS)
    event_type = 'honor' if is_honor else 'award'

    # Extract awarding body from parentheses
    paren_match = re.search(r'\(([^)]+)\)', cleaned)
    if paren_match:
        awarding_body = paren_match.group(1).split(',')[0].strip()
        award_name = cleaned[:cleaned.index('(')].strip()
    else:
        award_name = cleaned
        # Try to split: "Nth [institution] [award]" pattern (Korean source)
        awarding_body = None

    # National exhibition tracking
    national_info = None
    if NATIONAL_EXHIBITION_RE.search(entry):
        if '대상' in entry or '대통령상' in entry:
            national_info = 'grand_prize'
        elif '특선' in entry:
            national_info = 'special_selection'
        elif '입선' in entry:
            national_info = 'selected'

    inst_ko = awarding_body or award_name
    event = {
        'year': year,
        'event_type': event_type,
        'event_ko': f"{award_name} 수상",
        'event_en': None,
        'detail_ko': None,
        'detail_en': None,
        'institution_ko': inst_ko,
        'institution_en': inst_en_map.get(inst_ko, award_body_map.get(inst_ko, inst_ko)),
        'institution_type': inst_type_map.get(inst_ko, 'other'),
        'country': 'KR',
        'city_ko': None,
        'city_en': None,
    }

    award_record = {
        'name_ko': award_name,
        'name_en': award_name_map.get(award_name, award_name),
        'year': year,
        'awarding_body_ko': awarding_body,
        'awarding_body_en': award_body_map.get(awarding_body, awarding_body),
    }

    return event, award_record, national_info


def parse_position_entry(entry, inst_en_map, inst_type_map, inst_country_map,
                         position_map):
    """Parse a career/position entry line."""
    year = extract_year(entry)
    cleaned = re.sub(r'^\d{4}[년.]?\d{0,2}\s*', '', entry).strip()

    # Detect residency
    is_residency = any(kw in entry for kw in RESIDENCY_KEYWORDS)
    event_type = 'residency' if is_residency else 'position'

    # Try to extract institution
    institution = None
    city = None

    # Common pattern: "[institution] [position] appointment" (Korean source)
    paren_match = re.search(r'\(([^)]+)\)', cleaned)
    if paren_match:
        institution = paren_match.group(1).split(',')[0].strip()
        city_parts = paren_match.group(1).split(',')
        if len(city_parts) > 1:
            city = city_parts[-1].strip()
    else:
        # Try to find institution name (ends with university, museum, etc.)
        inst_match = re.search(
            r'([가-힣A-Za-z]+(?:대학교|대학|미술관|갤러리|재단|센터|학교|연구소|협회|위원회))',
            cleaned
        )
        if inst_match:
            institution = inst_match.group(1)

    country = detect_country(entry, city) if city else 'KR'
    if institution and institution in inst_country_map:
        country = inst_country_map[institution]

    inst_en = inst_en_map.get(institution, institution) if institution else None
    inst_type = inst_type_map.get(institution, 'other') if institution else 'other'

    return {
        'year': year,
        'event_type': event_type,
        'event_ko': cleaned,
        'event_en': None,
        'detail_ko': None,
        'detail_en': None,
        'institution_ko': institution,
        'institution_en': inst_en,
        'institution_type': inst_type,
        'country': country,
        'city_ko': city,
        'city_en': None,
    }


def parse_education_entries(entries, inst_en_map, inst_type_map, dept_map):
    """Parse education entries into structured education history."""
    education_history = []
    highest_degree = None
    degree_rank = {'phd': 4, 'master': 3, 'bachelor': 2, 'diploma': 1, 'other': 0}
    highest_rank = -1

    study_abroad = False
    study_abroad_countries = set()

    for entry in entries:
        degree = detect_degree(entry)
        rank = degree_rank.get(degree, 0)
        if rank > highest_rank:
            highest_rank = rank
            highest_degree = degree

        # Extract institution name
        inst_match = re.match(
            r'^([^\s]+(?:대학교|대학|학교|University|College|School|Institute))',
            entry
        )
        if inst_match:
            institution = inst_match.group(1)
        else:
            institution = entry.split()[0] if entry.split() else entry

        # Extract department
        dept = None
        remaining = entry[len(institution):].strip() if institution in entry else entry
        for deg_kw_list in DEGREE_KEYWORDS.values():
            for kw in deg_kw_list:
                idx = remaining.find(kw)
                if idx > 0:
                    dept = remaining[:idx].strip().rstrip(' 및')
                    break
            if dept:
                break
        if not dept:
            dept = remaining.split('졸업')[0].strip() if '졸업' in remaining else None

        # Determine if overseas
        is_overseas = False
        country = 'KR'
        for country_ko, code in COUNTRY_KO_TO_CODE.items():
            if country_ko in entry:
                is_overseas = True
                country = code
                break
        if institution in inst_type_map:
            if inst_type_map[institution] == 'overseas_university':
                is_overseas = True
                if country == 'KR':
                    country = 'JP'  # default for overseas_university if no explicit country

        if is_overseas:
            study_abroad = True
            study_abroad_countries.add(country)

        inst_type = inst_type_map.get(
            institution,
            'domestic_university' if country == 'KR' else 'overseas_university'
        )
        inst_en = inst_en_map.get(institution, institution)
        dept_en = dept_map.get(dept) if dept else None

        education_history.append({
            'institution_ko': institution,
            'institution_en': inst_en,
            'institution_type': inst_type,
            'department_ko': dept,
            'department_en': dept_en,
            'degree': degree,
            'graduation_year': None,
            'country': country,
        })

    primary_alma = education_history[0]['institution_ko'] if education_history else None
    primary_alma_en = education_history[0]['institution_en'] if education_history else None

    return {
        'highest_degree': highest_degree,
        'education_history': education_history,
        'study_abroad': study_abroad,
        'study_abroad_countries': sorted(study_abroad_countries),
        'primary_alma_mater_ko': primary_alma,
        'primary_alma_mater_en': primary_alma_en,
    }


def parse_collection_entries(entries, inst_en_map):
    """Parse collection entries (comma-separated institution names)."""
    collections = []
    for entry in entries:
        parts = re.split(r'[,，、]', entry)
        for part in parts:
            part = part.strip()
            if part and len(part) > 1:
                collections.append({
                    'institution_ko': part,
                    'institution_en': inst_en_map.get(part, part),
                })
    return collections


# ============================================================
# Summary computation
# ============================================================

def compute_generation_cohort(birth_year):
    """Compute generation cohort from birth year (decade-based)."""
    if birth_year is None:
        return None
    decade = (birth_year // 10) * 10
    return f"{decade}s"


def compute_career_timeline(career_events, birth_year):
    """Compute career timeline from events."""
    years = [e['year'] for e in career_events if e.get('year')]
    if not years:
        return {
            'first_solo_exhibition_year': None,
            'debut_age': None,
            'years_to_debut': None,
            'first_major_award_year': None,
            'first_mmca_exhibition_year': None,
            'first_overseas_exhibition_year': None,
            'career_span': {'start_year': None, 'end_year': None, 'total_years': 0},
        }

    solo_years = [e['year'] for e in career_events if e['event_type'] == 'solo_exhibition' and e.get('year')]
    award_years = [e['year'] for e in career_events if e['event_type'] == 'award' and e.get('year')]
    mmca_years = [e['year'] for e in career_events
                  if '국립현대미술관' in (e.get('institution_ko') or '') and e.get('year')]
    overseas_years = [e['year'] for e in career_events if e.get('country', 'KR') != 'KR' and e.get('year')]

    first_solo = min(solo_years) if solo_years else None
    first_award = min(award_years) if award_years else None
    first_mmca = min(mmca_years) if mmca_years else None
    first_overseas = min(overseas_years) if overseas_years else None

    start_year = min(years)
    end_year = max(years)
    debut_age = (first_solo - birth_year) if (first_solo and birth_year) else None

    return {
        'first_solo_exhibition_year': first_solo,
        'debut_age': debut_age,
        'years_to_debut': None,
        'first_major_award_year': first_award,
        'first_mmca_exhibition_year': first_mmca,
        'first_overseas_exhibition_year': first_overseas,
        'career_span': {
            'start_year': start_year,
            'end_year': end_year,
            'total_years': end_year - start_year + 1 if start_year and end_year else 0,
        },
    }


def compute_exhibition_metrics(career_events):
    """Compute exhibition summary metrics."""
    solo = [e for e in career_events if e['event_type'] == 'solo_exhibition']
    group = [e for e in career_events if e['event_type'] == 'group_exhibition']

    # Solo by decade
    solo_by_decade = {}
    for decade_start in range(1960, 2030, 10):
        label = f"{decade_start}s"
        solo_by_decade[label] = sum(
            1 for e in solo if decade_start <= (e.get('year') or 0) < decade_start + 10
        )

    domestic_solo = sum(1 for e in solo if e.get('country', 'KR') == 'KR')
    overseas_solo = sum(1 for e in solo if e.get('country', 'KR') != 'KR')
    overseas_countries = sorted(set(
        e.get('country') for e in solo if e.get('country') and e['country'] != 'KR'
    ))

    return {
        'total_solo_exhibitions': len(solo),
        'total_group_exhibitions': len(group),
        'solo_by_decade': solo_by_decade,
        'domestic_solo_count': domestic_solo,
        'overseas_solo_count': overseas_solo,
        'overseas_countries': overseas_countries,
    }


def compute_institutional_engagement(career_events, inst_en_map, inst_type_map):
    """Compute institutional engagement metrics."""
    mmca_events = [e for e in career_events
                   if '국립현대미술관' in (e.get('institution_ko') or '')]
    sema_events = [e for e in career_events
                   if '서울시립미술관' in (e.get('institution_ko') or '')]

    # Venue frequency
    venue_counter = Counter()
    for e in career_events:
        inst = e.get('institution_ko')
        if inst:
            venue_counter[inst] += 1

    frequent_venues = []
    for inst_ko, count in venue_counter.most_common(10):
        if count < 2:
            break
        frequent_venues.append({
            'name_ko': inst_ko,
            'name_en': inst_en_map.get(inst_ko, inst_ko),
            'type': inst_type_map.get(inst_ko, 'other'),
            'count': count,
        })

    # Major museums
    public_museum_counter = Counter()
    private_museum_counter = Counter()
    for e in career_events:
        inst_ko = e.get('institution_ko')
        itype = e.get('institution_type', '')
        if itype == 'public_museum' and inst_ko:
            public_museum_counter[inst_ko] += 1
        elif itype == 'private_museum' and inst_ko:
            private_museum_counter[inst_ko] += 1

    major_public_ko = [k for k, _ in public_museum_counter.most_common(10)]
    major_public_en = [inst_en_map.get(k, k) for k in major_public_ko]
    major_private_ko = [k for k, _ in private_museum_counter.most_common(10)]
    major_private_en = [inst_en_map.get(k, k) for k in major_private_ko]

    # Biennale participation
    biennale_events = [e for e in career_events if e['event_type'] == 'biennale']
    biennale_participation = []
    seen = set()
    for e in biennale_events:
        key = (e.get('institution_ko'), e.get('year'))
        if key not in seen:
            seen.add(key)
            biennale_participation.append({
                'name_ko': e.get('institution_ko') or e.get('event_ko'),
                'name_en': e.get('institution_en') or e.get('event_en'),
                'country': e.get('country', 'KR'),
                'year': e.get('year'),
            })

    return {
        'mmca_exhibited': len(mmca_events) > 0,
        'mmca_exhibition_count': len(mmca_events),
        'mmca_first_year': min((e['year'] for e in mmca_events if e.get('year')), default=None),
        'sema_exhibited': len(sema_events) > 0,
        'major_public_museums_ko': major_public_ko,
        'major_public_museums_en': major_public_en,
        'major_private_museums_ko': major_private_ko,
        'major_private_museums_en': major_private_en,
        'frequent_venues': frequent_venues,
        'biennale_participation': biennale_participation,
    }


def compute_overseas_activity(career_events):
    """Compute overseas activity summary."""
    overseas_events = [e for e in career_events if e.get('country', 'KR') != 'KR']
    has_overseas = len(overseas_events) > 0
    active_countries = sorted(set(e['country'] for e in overseas_events if e.get('country')))

    return {
        'has_overseas_experience': has_overseas,
        'active_countries': active_countries,
        'residencies': [],
        'art_fairs_ko': [],
        'art_fairs_en': [],
    }


def compute_collection_presence(collections):
    """Compute collection presence flags."""
    mmca_collected = any(
        '국립현대미술관' in (c.get('institution_ko') or '')
        for c in collections
    )
    major_collected = len(collections) > 0

    return {
        'is_collected_by_mmca': mmca_collected,
        'is_collected_by_major_museums': major_collected,
        'collections': collections,
        'total_collection_count': len(collections),
    }


# ============================================================
# Main pipeline
# ============================================================

def process_artist(raw, inst_en_map, inst_type_map, inst_country_map,
                   award_name_map, award_body_map, dept_map,
                   bp_map, cat_map, city_map, member_map, position_map):
    """Process a single raw artist record into enriched format."""
    career = raw.get('career', {})
    raw_text = career.get('raw_text') or ''

    # Basic info
    birth_year = int(raw['birth_year']) if raw.get('birth_year') else None
    death_year = int(raw['death_year']) if raw.get('death_year') else None
    birth_info = career.get('birth_info') or ''

    # Extract birth place from birth_info (or from raw_text first line)
    birth_place_ko = None
    bp_source = birth_info or ''
    if not bp_source and raw_text:
        first_line = raw_text.split('\n')[0].strip()
        if re.search(r'\d{4}', first_line) and '출생' in first_line:
            bp_source = first_line
    bp_match = re.search(r'\d{4}\s*(.+?)출생', bp_source) if bp_source else None
    if bp_match:
        birth_place_ko = bp_match.group(1).strip()
    birth_place_en = bp_map.get(birth_place_ko) if birth_place_ko else None

    # Category
    category_ko = raw.get('category', '')
    category_en = cat_map.get(category_ko, category_ko)
    primary_category = category_en.lower().replace(' ', '_') if category_en else None

    # ---- Parse raw_text into sections ----
    sections = split_raw_text_into_sections(raw_text)

    # If raw_text parsing yields empty sections, fall back to pre-split career data
    if not sections.get('solo_exhibition') and career.get('solo_exhibitions'):
        sections['solo_exhibition'] = career['solo_exhibitions']
    if not sections.get('group_exhibition') and career.get('group_exhibitions'):
        sections['group_exhibition'] = career['group_exhibitions']
    if not sections.get('award') and career.get('awards'):
        sections['award'] = career['awards']
    if not sections.get('education') and career.get('education'):
        sections['education'] = career['education']
    if not sections.get('collection') and career.get('collections'):
        sections['collection'] = career['collections']

    # ---- Parse each section ----
    all_events = []
    awards_list = []
    national_record = {'grand_prize': 0, 'special_selection': 0, 'selected': 0}
    honors = []

    # Solo exhibitions
    for entry in sections.get('solo_exhibition', []):
        evt = parse_exhibition_entry(
            entry, 'solo_exhibition',
            inst_en_map, inst_type_map, inst_country_map
        )
        if evt:
            all_events.append(evt)

    # Group exhibitions
    for entry in sections.get('group_exhibition', []):
        evt = parse_exhibition_entry(
            entry, 'group_exhibition',
            inst_en_map, inst_type_map, inst_country_map
        )
        if evt:
            all_events.append(evt)

    # Awards (includes honor detection)
    for entry in sections.get('award', []):
        result = parse_award_entry(
            entry, inst_en_map, inst_type_map,
            award_name_map, award_body_map
        )
        if result and result[0] is not None:
            evt, award_rec, nat_info = result
            all_events.append(evt)
            if award_rec:
                awards_list.append(award_rec)
            if nat_info:
                national_record[nat_info] += 1
            if evt['event_type'] == 'honor':
                honors.append(evt['event_ko'])

    # Positions / Career entries
    for entry in sections.get('position', []):
        evt = parse_position_entry(
            entry, inst_en_map, inst_type_map, inst_country_map,
            position_map
        )
        if evt:
            all_events.append(evt)

    # Education
    education_data = parse_education_entries(
        sections.get('education', []),
        inst_en_map, inst_type_map, dept_map
    )

    # Education events
    for eh in education_data['education_history']:
        all_events.append({
            'year': eh.get('graduation_year'),
            'event_type': 'education',
            'event_ko': f"{eh['institution_ko']} {eh.get('department_ko') or ''} {eh['degree']}".strip(),
            'event_en': None,
            'detail_ko': None,
            'detail_en': None,
            'institution_ko': eh['institution_ko'],
            'institution_en': eh['institution_en'],
            'institution_type': eh['institution_type'],
            'country': eh.get('country', 'KR'),
            'city_ko': None,
            'city_en': None,
        })

    # Collections
    collections = parse_collection_entries(
        sections.get('collection', []),
        inst_en_map
    )
    for c in collections:
        all_events.append({
            'year': None,
            'event_type': 'collection',
            'event_ko': f"작품 소장: {c['institution_ko']}",
            'event_en': None,
            'detail_ko': None,
            'detail_en': None,
            'institution_ko': c['institution_ko'],
            'institution_en': c['institution_en'],
            'institution_type': inst_type_map.get(c['institution_ko'], 'other'),
            'country': inst_country_map.get(c['institution_ko'], 'KR'),
            'city_ko': None,
            'city_en': None,
        })

    # Sort by year
    all_events.sort(key=lambda e: (e.get('year') or 9999, e.get('event_type', '')))

    # Apply city translations
    for e in all_events:
        if e.get('city_ko') and not e.get('city_en'):
            e['city_en'] = city_map.get(e['city_ko'])

    # ---- Compute summaries ----
    career_timeline = compute_career_timeline(all_events, birth_year)
    exhibition_metrics = compute_exhibition_metrics(all_events)
    inst_engagement = compute_institutional_engagement(
        all_events, inst_en_map, inst_type_map
    )
    overseas_activity = compute_overseas_activity(all_events)
    collection_presence = compute_collection_presence(collections)

    # Professional roles: extract from position events
    academic_positions = []
    is_professor = False
    jury_experience = False
    jury_positions_ko = []
    memberships_ko = []

    for evt in all_events:
        if evt['event_type'] == 'position':
            eko = evt.get('event_ko', '')
            # Academic positions
            if any(kw in eko for kw in ['교수', '조교수', '부교수', '강사', '초빙']):
                is_professor = True
                academic_positions.append({
                    'institution_ko': evt.get('institution_ko'),
                    'institution_en': evt.get('institution_en'),
                    'position_ko': eko,
                    'position_en': position_map.get(eko),
                    'start_year': evt.get('year'),
                    'end_year': None,
                })
            # Jury/committee
            if any(kw in eko for kw in ['심사위원', '운영위원', '심의위원', '위촉']):
                jury_experience = True
                jury_positions_ko.append(eko)
            # Memberships
            if any(kw in eko for kw in ['회원', '회장', '이사', '협회']):
                memberships_ko.append(eko)

    professional_roles = {
        'academic_positions': academic_positions,
        'is_professor': is_professor,
        'jury_experience': jury_experience,
        'jury_positions_ko': jury_positions_ko,
        'jury_positions_en': [position_map.get(p, p) for p in jury_positions_ko],
        'memberships_ko': memberships_ko,
        'memberships_en': [member_map.get(m, m) for m in memberships_ko],
    }

    # Build final record
    record = {
        'metadata': {
            'artist_id': raw['id'],
            'source_url': raw.get('url', ''),
        },
        'basic_info': {
            'name_ko': raw.get('name_ko', ''),
            'name_en': raw.get('name_en', ''),
            'name_hanja': raw.get('name_hanja'),
            'birth_year': birth_year,
            'death_year': death_year,
            'birth_place_ko': birth_place_ko,
            'birth_place_en': birth_place_en,
            'generation_cohort': compute_generation_cohort(birth_year),
        },
        'education': education_data,
        'career_timeline': career_timeline,
        'exhibition_metrics': exhibition_metrics,
        'institutional_engagement': inst_engagement,
        'awards_recognition': {
            'awards_list': awards_list,
            'national_exhibition_record': national_record,
            'honors': honors,
        },
        'collection_presence': collection_presence,
        'professional_roles': professional_roles,
        'artistic_profile': {
            'primary_category': primary_category,
            'primary_category_ko': category_ko,
            'primary_category_en': category_en,
            'secondary_categories': [],
        },
        'overseas_activity': overseas_activity,
        'career_events': all_events,
    }

    return record


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    # Load inputs
    print("Loading raw data...")
    raw_data = load_json(os.path.join(data_dir, 'artist_details.json'))
    print(f"  {len(raw_data)} artists loaded from artist_details.json")

    # Load canonical mappings
    print("Loading canonical mappings...")
    inst_maps = load_json(os.path.join(data_dir, 'institution_mappings.json'))
    trans_maps = load_json(os.path.join(data_dir, 'translation_mappings.json'))

    inst_en_map = inst_maps['institution_ko_to_en']
    inst_type_map = inst_maps['institution_ko_to_type']
    inst_country_map = inst_maps['institution_ko_to_country']

    award_name_map = trans_maps['award_name_ko_to_en']
    award_body_map = trans_maps['award_body_ko_to_en']
    dept_map = trans_maps['department_ko_to_en']
    bp_map = trans_maps['birth_place_ko_to_en']
    cat_map = trans_maps['category_ko_to_en']
    city_map = trans_maps['city_ko_to_en']
    member_map = trans_maps['membership_ko_to_en']
    position_map = trans_maps['position_ko_to_en']

    print(f"  Institution en: {len(inst_en_map)}, type: {len(inst_type_map)}, country: {len(inst_country_map)}")
    print(f"  Awards: {len(award_name_map)}, depts: {len(dept_map)}, cities: {len(city_map)}")

    # Process each artist
    print("Processing artists...")
    results = []
    processed_ids = []
    for i, raw in enumerate(raw_data):
        record = process_artist(
            raw, inst_en_map, inst_type_map, inst_country_map,
            award_name_map, award_body_map, dept_map,
            bp_map, cat_map, city_map, member_map, position_map
        )
        results.append(record)
        processed_ids.append(raw['id'])
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(raw_data)} artists...")

    # Build output
    output = {
        'processed_ids': processed_ids,
        'results': results,
        'stats': {
            'processed': len(results),
            'success': len(results),
            'failed': 0,
            'skipped': 0,
        }
    }

    # Save to a separate file to avoid overwriting the canonical data.json.
    # Use --overwrite flag to write directly to data.json.
    if '--overwrite' in sys.argv:
        output_path = os.path.join(data_dir, 'data.json')
    else:
        output_path = os.path.join(data_dir, 'data_rebuilt.json')
    save_json(output, output_path)
    print(f"\nSaved {len(results)} artists to {output_path}")
    if output_path.endswith('data_rebuilt.json'):
        print("  (Use --overwrite to write directly to data.json)")

    # Verification
    total_events = sum(len(r['career_events']) for r in results)
    with_events = sum(1 for r in results if r['career_events'])
    print(f"  Total career events: {total_events}")
    print(f"  Artists with events: {with_events}/{len(results)}")


if __name__ == '__main__':
    main()
