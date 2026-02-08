#!/usr/bin/env python3
"""
DA-Arts artist 500 crawler main script.

Usage:
    python main.py                    # Full crawl
    python main.py --list-only        # List only
    python main.py --detail-only      # Detail only (requires list file)
    python main.py --test             # Test mode (5 artists)
    python main.py --resume           # Resume previous crawl
"""

import argparse
import json
import os
import sys
from datetime import datetime

from crawler.artist_list_crawler import ArtistListCrawler
from crawler.artist_detail_crawler import ArtistDetailCrawler


def setup_directories():
    """Create required directories"""
    dirs = ['data', 'data/raw', 'data/checkpoints', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def crawl_artist_list(delay: float = 1.0) -> list:
    """Crawl artist list"""
    print("=" * 60)
    print("Step 1: Artist list crawl")
    print("=" * 60)
    
    crawler = ArtistListCrawler()
    artists = crawler.crawl_all(delay=delay)
    crawler.save_to_json('data/artist_list.json')
    
    return artists


def crawl_artist_details(artists: list, delay: float = 1.5, 
                         max_workers: int = 1, resume: bool = False) -> list:
    """Crawl artist details"""
    print("\n" + "=" * 60)
    print("Step 2: Artist detail crawl")
    print("=" * 60)
    
    estimated_time = len(artists) * delay / 60
    print(f"Target artists: {len(artists)}")
    print(f"Estimated time: ~{estimated_time:.1f} min")
    print("=" * 60)
    
    existing_results = []
    crawled_ids = set()
    
    if resume and os.path.exists('data/artist_details.json'):
        try:
            with open('data/artist_details.json', 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                crawled_ids = {r['id'] for r in existing_results if r.get('id')}
                print(f"Loaded previous results: {len(existing_results)}")
        except Exception as e:
            print(f"Failed to load previous results: {e}")
    
    if crawled_ids:
        remaining_artists = [a for a in artists if a.get('id') not in crawled_ids]
        print(f"Remaining artists: {len(remaining_artists)}")
    else:
        remaining_artists = artists
    
    if not remaining_artists:
        print("All artists already crawled.")
        return existing_results
    
    crawler = ArtistDetailCrawler()
    new_results = crawler.crawl_all(remaining_artists, delay=delay, max_workers=max_workers)
    
    all_results = existing_results + new_results
    crawler.results = all_results
    
    crawler.save_to_json('data/artist_details.json')
    
    crawler.save_to_csv('data/artist_details.csv')
    
    return all_results


def load_artist_list(filepath: str = 'data/artist_list.json') -> list:
    """Load saved artist list"""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run --list-only first.")
        sys.exit(1)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_report(artists: list, details: list):
    """Generate crawling report"""
    print("\n" + "=" * 60)
    print("Crawling Report")
    print("=" * 60)
    
    print(f"Total artists: {len(artists)}")
    print(f"Details collected: {len(details)}")
    print(f"Collection rate: {len(details)/len(artists)*100:.1f}%")
    
    categories = {}
    for artist in artists:
        cat = artist.get('category', 'Uncategorized')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nArtists by category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  - {cat}: {count}")
    
    has_education = sum(1 for d in details if d.get('career', {}).get('education'))
    has_solo = sum(1 for d in details if d.get('career', {}).get('solo_exhibitions'))
    has_group = sum(1 for d in details if d.get('career', {}).get('group_exhibitions'))
    has_awards = sum(1 for d in details if d.get('career', {}).get('awards'))
    
    print("\nCareer info coverage:")
    print(f"  - Education: {has_education}")
    print(f"  - Solo exhibitions: {has_solo}")
    print(f"  - Group exhibitions: {has_group}")
    print(f"  - Awards: {has_awards}")
    
    report = {
        'crawled_at': datetime.now().isoformat(),
        'total_artists': len(artists),
        'details_collected': len(details),
        'collection_rate': len(details) / len(artists) * 100,
        'categories': categories,
        'career_stats': {
            'education': has_education,
            'solo_exhibitions': has_solo,
            'group_exhibitions': has_group,
            'awards': has_awards,
        }
    }
    
    with open('data/crawling_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved: data/crawling_report.json")


def main():
    parser = argparse.ArgumentParser(
        description='DA-Arts artist 500 crawler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Full crawl (~505 artists, ~15 min)
    python main.py --list-only        # List only
    python main.py --detail-only      # Detail only
    python main.py --resume           # Resume previous crawl
    python main.py --test             # Test mode (5 artists)
    python main.py --delay 2.0        # 2 sec between requests
        """
    )
    
    parser.add_argument('--list-only', action='store_true',
                        help='Crawl list only')
    parser.add_argument('--detail-only', action='store_true',
                        help='Crawl details only (requires list file)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume previous crawl')
    parser.add_argument('--test', action='store_true',
                        help='Test mode (5 artists only)')
    parser.add_argument('--delay', type=float, default=1.5,
                        help='Request delay in seconds (default: 1.5)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    
    args = parser.parse_args()
    
    setup_directories()
    
    print("=" * 60)
    print("DA-Arts artist 500 crawler")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.resume:
        print("Mode: Resume previous crawl")
    print("=" * 60)
    
    artists = []
    details = []
    
    try:
        if args.detail_only or args.resume:
            artists = load_artist_list()
            if args.test:
                artists = artists[:5]
            details = crawl_artist_details(artists, delay=args.delay, 
                                          max_workers=args.workers,
                                          resume=args.resume)
        elif args.list_only:
            artists = crawl_artist_list(delay=args.delay)
        else:
            artists = crawl_artist_list(delay=args.delay)
            if args.test:
                artists = artists[:5]
            details = crawl_artist_details(artists, delay=args.delay,
                                          max_workers=args.workers,
                                          resume=False)
        
        if artists and details:
            generate_report(artists, details)
        
        print("\n" + "=" * 60)
        print("Crawl complete!")
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nCrawl interrupted by user.")
        print("Resume with: python main.py --resume")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
