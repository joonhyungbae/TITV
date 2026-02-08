"""
DA-Arts artist 500 detail crawler.
Collects career information from individual artist pages.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
from tqdm import tqdm
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ArtistDetailCrawler:
    """Artist detail information crawler"""
    
    BASE_URL = "https://www.daarts.or.kr"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.daarts.or.kr/visual/artist',
        })
        self.lock = threading.Lock()
        self.results: List[Dict] = []
    
    def parse_career_section(self, soup: BeautifulSoup) -> Dict:
        """Parse career section - tailored to DA-Arts website structure"""
        career = {
            'birth_info': None,
            'education': [],
            'solo_exhibitions': [],
            'group_exhibitions': [],
            'awards': [],
            'collections': [],
            'other_activities': [],
            'raw_text': None
        }
        
        try:
            # Find career section (DA-Arts structure: h3.stitle followed by div.scont.article)
            stitle = soup.find('h3', class_='stitle', string=lambda t: t and '약력' in str(t))
            
            if stitle:
                career_div = stitle.find_next_sibling('div', class_='scont')
                if not career_div:
                    career_div = stitle.find_next_sibling('div', class_='article')
                if not career_div:
                    for sibling in stitle.find_next_siblings('div'):
                        if 'scont' in sibling.get('class', []) or 'article' in sibling.get('class', []):
                            career_div = sibling
                            break
                
                if career_div:
                    raw_text = career_div.get_text(separator='\n', strip=True)
                    career['raw_text'] = raw_text
                    career = self._parse_career_text(raw_text, career)
            
            if not career['raw_text']:
                focus_wrap = soup.find('div', class_='focus_wrap')
                if focus_wrap:
                    scont = focus_wrap.find('div', class_='scont')
                    if scont:
                        raw_text = scont.get_text(separator='\n', strip=True)
                        career['raw_text'] = raw_text
                        career = self._parse_career_text(raw_text, career)
            
        except Exception as e:
            print(f"Career parse error: {e}")
        
        return career
    
    def _parse_career_text(self, text: str, career: Dict) -> Dict:
        """Parse career text into sections"""
        if not text:
            return career
        
        # Section markers (Korean keywords from DA-Arts website)
        section_markers = {
            '개인전': 'solo_exhibitions',
            '단체전': 'group_exhibitions',
            '그룹전': 'group_exhibitions',
            '수상': 'awards',
            '작품소장': 'collections',
            '소장처': 'collections',
            '기타활동': 'other_activities',
        }
        
        lines = text.split('\n')
        current_section = None
        current_items = []
        
        header_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            found_marker = False
            for marker, section_key in section_markers.items():
                if marker in line and len(line) < 20:
                    if current_section and current_items:
                        career[current_section].extend(current_items)
                    
                    current_section = section_key
                    current_items = []
                    found_marker = True
                    break
            
            if found_marker:
                continue
            
            if current_section:
                year_match = re.match(r'^(\d{4})\s+(.+)$', line)
                if year_match:
                    year = year_match.group(1)
                    content = year_match.group(2).strip()
                    current_items.append(f"{year}년 {content}")
                elif line and not line.startswith('전체'):
                    current_items.append(line)
            else:
                if i < 10:
                    header_lines.append(line)
        
        if current_section and current_items:
            career[current_section].extend(current_items)
        
        if header_lines:
            header_text = ' '.join(header_lines)
            birth_match = re.search(r'(\d{4})\s*(서울|부산|대구|인천|광주|대전|울산|경기|강원|충북|충남|전북|전남|경북|경남|제주|[가-힣]+)?\s*출생', header_text)
            if birth_match:
                career['birth_info'] = birth_match.group(0)
            edu_patterns = [
                r'([가-힣]+대학교?[가-힣\s]*(?:졸업|수료|재학|학사|석사|박사)[가-힣\s]*)',
                r'([가-힣]+대학[가-힣\s]*(?:및 동|동)[가-힣\s]*졸업)',
            ]
            for pattern in edu_patterns:
                edu_matches = re.findall(pattern, header_text)
                for match in edu_matches:
                    if match not in career['education']:
                        career['education'].append(match.strip())
        
        return career
    
    def parse_basic_info(self, soup: BeautifulSoup) -> Dict:
        """Parse basic artist information"""
        info = {
            'name_ko': None,
            'name_en': None,
            'name_hanja': None,
            'birth_year': None,
            'death_year': None,
            'birth_place': None,
            'category': None,
            'description': None,
        }
        
        try:
            title = soup.find('h1') or soup.find('h2') or soup.find('.artist-name')
            if title:
                title_text = title.get_text(strip=True)
                name_match = re.search(r'([가-힣]+)\s*\(([^)]+)\)', title_text)
                if name_match:
                    info['name_ko'] = name_match.group(1)
                    info['name_en'] = name_match.group(2)
                else:
                    info['name_ko'] = title_text
            
            meta_selectors = [
                '.artist-meta',
                '.profile-info',
                '.basic-info',
                '.artist-info',
            ]
            
            for selector in meta_selectors:
                meta = soup.select_one(selector)
                if meta:
                    meta_text = meta.get_text()
                    birth_match = re.search(r'(\d{4})\s*년?\s*생', meta_text)
                    if birth_match:
                        info['birth_year'] = birth_match.group(1)
                    death_match = re.search(r'(\d{4})\s*년?\s*몰', meta_text)
                    if death_match:
                        info['death_year'] = death_match.group(1)
                    place_match = re.search(r'출생지[:\s]*([^\n,]+)', meta_text)
                    if place_match:
                        info['birth_place'] = place_match.group(1).strip()
                    
                    break
            
            categories = ['회화', '조각', '공예', '사진', '판화', '설치', '영상', '디자인', '서예', '미디어']
            full_text = soup.get_text()
            for cat in categories:
                if cat in full_text[:500]:
                    info['category'] = cat
                    break
            
            desc_selectors = [
                '.artist-description',
                '.profile-description',
                '.intro',
                '.description',
                'p.intro',
            ]
            
            for selector in desc_selectors:
                desc = soup.select_one(selector)
                if desc:
                    info['description'] = desc.get_text(strip=True)[:1000]
                    break
            
        except Exception as e:
            print(f"Basic info parse error: {e}")
        
        return info
    
    def parse_works(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse artwork information"""
        works = []
        
        try:
            work_selectors = [
                '.work-item',
                '.artwork-item',
                '.gallery-item',
                '.work-list li',
            ]
            
            for selector in work_selectors:
                items = soup.select(selector)
                if items:
                    for item in items[:20]:
                        work = {
                            'title': None,
                            'year': None,
                            'medium': None,
                            'size': None,
                            'image_url': None,
                        }
                        title_el = item.select_one('.title, .work-title, h3, h4')
                        if title_el:
                            work['title'] = title_el.get_text(strip=True)
                        img = item.select_one('img')
                        if img:
                            work['image_url'] = img.get('src') or img.get('data-src')
                        year_match = re.search(r'(\d{4})', item.get_text())
                        if year_match:
                            work['year'] = year_match.group(1)
                        
                        if work['title'] or work['image_url']:
                            works.append(work)
                    break
            
        except Exception as e:
            print(f"Works parse error: {e}")
        
        return works
    
    def crawl_artist(self, artist: Dict, retry: int = 3) -> Optional[Dict]:
        """Crawl individual artist detail information"""
        url = artist.get('url')
        if not url:
            return None
        
        for attempt in range(retry):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'lxml')
                basic_info = self.parse_basic_info(soup)
                career = self.parse_career_section(soup)
                works = self.parse_works(soup)
                result = {
                    'id': artist.get('id'),
                    'url': url,
                    **basic_info,
                    'career': career,
                    'works': works,
                    'raw_html_length': len(response.text),
                }
                if not result['name_ko']:
                    result['name_ko'] = artist.get('name_ko')
                if not result['name_en']:
                    result['name_en'] = artist.get('name_en')
                if not result['birth_year']:
                    result['birth_year'] = artist.get('birth_year')
                if not result['category']:
                    result['category'] = artist.get('category')
                
                return result
                
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{retry}): {url} - {e}")
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"Parse failed: {url} - {e}")
                break
        
        return None
    
    def crawl_all(self, artists: List[Dict], delay: float = 1.0, 
                  max_workers: int = 1, save_interval: int = 50) -> List[Dict]:
        """Crawl all artist detail information"""
        results = []
        failed = []
        
        print(f"Starting detail crawl for {len(artists)} artists...")
        
        if max_workers == 1:
            for i, artist in enumerate(tqdm(artists, desc="Crawling artists")):
                result = self.crawl_artist(artist)
                if result:
                    results.append(result)
                else:
                    failed.append(artist)
                if (i + 1) % save_interval == 0:
                    self._save_intermediate(results, i + 1)
                
                time.sleep(delay)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.crawl_artist, artist): artist 
                          for artist in artists}
                
                completed = 0
                for future in tqdm(as_completed(futures), total=len(artists), 
                                  desc="Crawling artists"):
                    result = future.result()
                    if result:
                        with self.lock:
                            results.append(result)
                    completed += 1
                    if completed % save_interval == 0:
                        with self.lock:
                            self._save_intermediate(results, completed)
                    
                    time.sleep(delay / max_workers)
        
        self.results = results
        print(f"\nCollected detail info for {len(results)} artists")
        if failed:
            print(f"Failed: {len(failed)} artists")
            with open('data/failed_artists.json', 'w', encoding='utf-8') as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
        
        return results
    
    def _save_intermediate(self, results: List[Dict], count: int):
        """Save intermediate results for recovery on interrupt"""
        filepath = 'data/artist_details.json'
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        tqdm.write(f"[Auto-save] {count} done, {filepath}")
    
    def save_to_json(self, filepath: str = 'data/artist_details.json'):
        """Save collected detail information to JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"Artist details saved: {filepath}")
    
    def save_to_csv(self, filepath: str = 'data/artist_details.csv'):
        """Save collected detail information to CSV"""
        import pandas as pd
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        flat_data = []
        for artist in self.results:
            row = {
                'id': artist.get('id'),
                'name_ko': artist.get('name_ko'),
                'name_en': artist.get('name_en'),
                'birth_year': artist.get('birth_year'),
                'death_year': artist.get('death_year'),
                'birth_place': artist.get('birth_place'),
                'category': artist.get('category'),
                'description': artist.get('description'),
                'url': artist.get('url'),
                'education': '; '.join(artist.get('career', {}).get('education', [])),
                'solo_exhibitions': '; '.join(artist.get('career', {}).get('solo_exhibitions', [])),
                'group_exhibitions': '; '.join(artist.get('career', {}).get('group_exhibitions', [])),
                'awards': '; '.join(artist.get('career', {}).get('awards', [])),
                'collections': '; '.join(artist.get('career', {}).get('collections', [])),
                'other_activities': '; '.join(artist.get('career', {}).get('other_activities', [])),
            }
            flat_data.append(row)
        
        df = pd.DataFrame(flat_data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"Artist details CSV saved: {filepath}")


if __name__ == "__main__":
    crawler = ArtistDetailCrawler()
    
    test_artist = {
        'id': '19200',
        'name_ko': '강경구',
        'name_en': 'Kang kyung koo',
        'url': 'https://www.daarts.or.kr/handle/11080/19200'
    }
    
    result = crawler.crawl_artist(test_artist)
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
