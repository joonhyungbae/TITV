"""
DA-Arts artist 500 list crawler.
Collects artist URLs and basic info from all pages.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from tqdm import tqdm
from typing import List, Dict, Optional


class ArtistListCrawler:
    """Artist list page crawler"""
    
    BASE_URL = "https://www.daarts.or.kr"
    LIST_URL = "https://www.daarts.or.kr/visual/artist"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.daarts.or.kr/',
        })
        self.artists: List[Dict] = []
    
    def get_total_pages(self) -> int:
        """Calculate total number of pages"""
        try:
            response = self.session.get(self.LIST_URL, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            full_text = soup.get_text()
            total_match = re.search(r'전체\s*_?(\d+)_?\s*명', full_text)
            if total_match:
                total_artists = int(total_match.group(1))
                pages = (total_artists + 11) // 12
                print(f"Total artists: {total_artists}, expected pages: {pages}")
                return pages
            
            for elem in soup.find_all(['span', 'strong', 'em', 'p']):
                text = elem.get_text(strip=True)
                match = re.search(r'(\d+)\s*명', text)
                if match and '전체' in text:
                    total_artists = int(match.group(1))
                    pages = (total_artists + 11) // 12
                    print(f"Total artists: {total_artists}, expected pages: {pages}")
                    return pages
            
            last_page_link = soup.select_one('a[href*="page="]')
            if last_page_link:
                page_links = soup.select('a[href*="page="]')
                max_page = 1
                for link in page_links:
                    href = link.get('href', '')
                    page_match = re.search(r'page=(\d+)', href)
                    if page_match:
                        max_page = max(max_page, int(page_match.group(1)))
                if max_page > 1:
                    print(f"Found from pagination: {max_page} pages")
                    return max_page
            
            print("Could not auto-detect page count. Using default 43 pages")
            return 43
            
        except Exception as e:
            print(f"Total page count failed: {e}")
            return 43
    
    def parse_artist_card(self, card) -> Optional[Dict]:
        """Extract info from artist card"""
        try:
            link = card.get('href', '')
            if not link:
                link_tag = card.find('a')
                if link_tag:
                    link = link_tag.get('href', '')
            
            if not link or 'handle' not in link:
                return None
            if link.startswith('/'):
                full_url = self.BASE_URL + link
            elif link.startswith('http'):
                full_url = link
            else:
                full_url = self.BASE_URL + '/' + link
            id_match = re.search(r'/handle/\d+/(\d+)', link)
            artist_id = id_match.group(1) if id_match else None
            text = card.get_text(separator=' ', strip=True)
            name_match = re.search(r'([가-힣]+)\s*\(([^)]+)\)', text)
            name_ko = name_match.group(1) if name_match else None
            name_en = name_match.group(2) if name_match else None
            birth_match = re.search(r'(\d{4})년생', text)
            birth_year = birth_match.group(1) if birth_match else None
            categories = ['회화', '조각', '공예', '사진', '판화', '설치', '영상', '디자인', '서예']
            category = None
            for cat in categories:
                if cat in text:
                    category = cat
                    break
            
            return {
                'id': artist_id,
                'name_ko': name_ko,
                'name_en': name_en,
                'birth_year': birth_year,
                'category': category,
                'url': full_url
            }
            
        except Exception as e:
            print(f"Artist card parse failed: {e}")
            return None
    
    def crawl_page(self, page: int) -> List[Dict]:
        """Collect artist list from a page"""
        artists = []
        
        try:
            if page == 1:
                url = self.LIST_URL
            else:
                url = f"{self.LIST_URL}?page={page}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            selectors = [
                'a[href*="handle/11080"]',
                '.artist-item a',
                '.list-item a',
                'ul.artist-list li a',
                '.content-list a[href*="handle"]',
            ]
            
            cards = []
            for selector in selectors:
                cards = soup.select(selector)
                if cards:
                    break
            if not cards:
                cards = soup.find_all('a', href=re.compile(r'/handle/\d+/\d+'))
            
            for card in cards:
                artist = self.parse_artist_card(card)
                if artist and artist.get('id'):
                    if not any(a['id'] == artist['id'] for a in artists):
                        artists.append(artist)
            
        except requests.exceptions.RequestException as e:
            print(f"Page {page} request failed: {e}")
        except Exception as e:
            print(f"Page {page} crawl failed: {e}")
        
        return artists
    
    def crawl_all(self, delay: float = 1.0) -> List[Dict]:
        """Collect artist list from all pages"""
        total_pages = self.get_total_pages()
        print(f"Starting crawl of {total_pages} pages...")
        
        all_artists = []
        seen_ids = set()
        
        for page in tqdm(range(1, total_pages + 1), desc="Crawling pages"):
            artists = self.crawl_page(page)
            
            for artist in artists:
                if artist['id'] not in seen_ids:
                    seen_ids.add(artist['id'])
                    all_artists.append(artist)
            
            time.sleep(delay)
        
        self.artists = all_artists
        print(f"Collected info for {len(all_artists)} artists")
        
        return all_artists
    
    def save_to_json(self, filepath: str = 'data/artist_list.json'):
        """Save collected artist list to JSON"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.artists, f, ensure_ascii=False, indent=2)
        
        print(f"Artist list saved: {filepath}")


if __name__ == "__main__":
    crawler = ArtistListCrawler()
    artists = crawler.crawl_all(delay=1.0)
    crawler.save_to_json()
