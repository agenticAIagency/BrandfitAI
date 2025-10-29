"""
Instagram Reel Analytics Scraper
Collects analytics data (likes, views, comments, etc.) for Instagram reels
Requires login to avoid 403 errors
"""

import instaloader
import time
import json
from pathlib import Path
from datetime import datetime

def scrape_analytics_programmatic(reel_urls, creator_name, username, password):
    """
    Programmatic version - can be called from other scripts
    Same as main() but accepts parameters instead of input()
    """
    return get_reel_analytics(reel_urls, creator_name, username, password)

def run_scraper(reel_urls, creator_name, username, password):
    """Wrapper to call from Agent 1"""
    return get_reel_analytics(reel_urls, creator_name, username, password)


def get_reel_analytics(reel_urls, creator_name, username=None, password=None):
    """
    Scrape analytics data from Instagram reels using Instaloader

    Args:
        reel_urls: List of Instagram reel URLs
        creator_name: Name for organizing output files
        username: Instagram username for login (recommended to avoid 403 errors)
        password: Instagram password
    """
    # Create Instaloader instance
    L = instaloader.Instaloader(
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern='',
        max_connection_attempts=3
    )

    # Login if credentials provided
    if username and password:
        print(f"\nLogging in as @{username}...")
        try:
            L.login(username, password)
            print("Login successful!\n")
        except instaloader.exceptions.BadCredentialsException:
            print("Login failed - bad credentials")
            return []
        except instaloader.exceptions.TwoFactorAuthRequiredException:
            print("Two-factor authentication required")
            return []
        except Exception as e:
            print(f"Login error: {e}")
            return []
    else:
        print("Warning: Running without login may result in 403 errors")
        print("Consider providing Instagram credentials for better results\n")

    # Create output folder
    output_folder = Path(f"analytics/{creator_name}")
    output_folder.mkdir(parents=True, exist_ok=True)

    analytics_data = []
    success_count = 0

    print(f"{'='*70}")
    print(f"Scraping Analytics for {len(reel_urls)} reels from @{creator_name}")
    print(f"{'='*70}\n")

    for idx, reel_url in enumerate(reel_urls, 1):
        print(f"[{idx}/{len(reel_urls)}] Processing: {reel_url}")

        # Extract shortcode from URL
        shortcode = reel_url.rstrip('/').split('/')[-1]

        try:
            # Get post data with retry logic
            post = None
            retry_count = 0
            max_retries = 2

            while retry_count < max_retries and post is None:
                try:
                    post = instaloader.Post.from_shortcode(L.context, shortcode)
                    break
                except instaloader.exceptions.ConnectionException as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"  Connection error, retrying ({retry_count}/{max_retries})...")
                        time.sleep(3)
                    else:
                        raise

            if post is None:
                raise Exception("Failed to fetch post after retries")

            # Safely extract tagged users
            tagged_usernames = []
            try:
                if hasattr(post, 'tagged_users'):
                    for user in post.tagged_users:
                        if hasattr(user, 'username'):
                            tagged_usernames.append(user.username)
                        elif isinstance(user, str):
                            tagged_usernames.append(user)
            except:
                pass

            # Safely extract location without triggering API call
            location_name = None
            try:
                if hasattr(post, '_location') and post._location:
                    location_name = post._location.get('name', None)
            except:
                pass

            # Safely extract caption
            caption_text = ""
            try:
                caption_text = post.caption if post.caption else ""
            except:
                pass

            # Safely extract hashtags and mentions
            hashtags_list = []
            mentions_list = []
            try:
                hashtags_list = list(post.caption_hashtags) if post.caption_hashtags else []
            except:
                pass
            try:
                mentions_list = list(post.caption_mentions) if post.caption_mentions else []
            except:
                pass

            # Extract all available analytics
            analytics = {
                'reel_url': reel_url,
                'shortcode': shortcode,
                'owner_username': post.owner_username,
                'owner_id': post.owner_id,
                'date_posted': post.date_utc.isoformat(),
                'caption': caption_text,
                'likes': post.likes,
                'comments': post.comments,
                'video_view_count': post.video_view_count if post.is_video else None,
                'video_duration': post.video_duration if post.is_video else None,
                'is_video': post.is_video,
                'typename': post.typename,
                'location': location_name,
                'hashtags': hashtags_list,
                'mentions': mentions_list,
                'tagged_users': tagged_usernames,
                'scraped_at': datetime.now().isoformat(),
                'success': True
            }

            # Calculate engagement metrics
            if analytics['video_view_count'] and analytics['video_view_count'] > 0:
                analytics['like_rate'] = round((analytics['likes'] / analytics['video_view_count']) * 100, 2)
                analytics['comment_rate'] = round((analytics['comments'] / analytics['video_view_count']) * 100, 2)
                analytics['engagement_rate'] = round(((analytics['likes'] + analytics['comments']) / analytics['video_view_count']) * 100, 2)
            else:
                analytics['like_rate'] = 0
                analytics['comment_rate'] = 0
                analytics['engagement_rate'] = 0

            analytics_data.append(analytics)
            success_count += 1

            # Print summary
            views_str = f"{analytics['video_view_count']:,}" if analytics['video_view_count'] else "N/A"
            print(f"  Likes: {analytics['likes']:,} | Views: {views_str} | Comments: {analytics['comments']:,}")
            if analytics.get('engagement_rate'):
                print(f"    Engagement Rate: {analytics['engagement_rate']}%")

        except instaloader.exceptions.QueryReturnedNotFoundException:
            print(f"  Post not found or deleted")
            analytics_data.append({
                'reel_url': reel_url,
                'shortcode': shortcode,
                'success': False,
                'error': 'Post not found or deleted',
                'scraped_at': datetime.now().isoformat()
            })

        except instaloader.exceptions.ConnectionException as e:
            error_msg = str(e)
            if '403' in error_msg:
                print(f"  Access denied (403) - Login required or rate limited")
            else:
                print(f"  Connection error: {e}")
            analytics_data.append({
                'reel_url': reel_url,
                'shortcode': shortcode,
                'success': False,
                'error': error_msg,
                'scraped_at': datetime.now().isoformat()
            })

        except Exception as e:
            print(f"  Unexpected error: {e}")
            analytics_data.append({
                'reel_url': reel_url,
                'shortcode': shortcode,
                'success': False,
                'error': str(e),
                'scraped_at': datetime.now().isoformat()
            })

        # Delay to avoid rate limiting
        if idx < len(reel_urls):
            delay = 5 if username else 8  # Longer delay if not logged in
            print(f"  Waiting {delay} seconds...\n")
            time.sleep(delay)

    # Calculate overall statistics
    successful_analytics = [a for a in analytics_data if a.get('success')]

    if successful_analytics:
        total_likes = sum(a.get('likes', 0) for a in successful_analytics)
        total_views = sum(a.get('video_view_count', 0) for a in successful_analytics if a.get('video_view_count'))
        total_comments = sum(a.get('comments', 0) for a in successful_analytics)
        avg_likes = total_likes / len(successful_analytics)
        avg_views = total_views / len(successful_analytics) if total_views > 0 else 0
        avg_comments = total_comments / len(successful_analytics)

        summary = {
            'creator_name': creator_name,
            'total_reels_analyzed': len(reel_urls),
            'successful_scrapes': success_count,
            'failed_scrapes': len(reel_urls) - success_count,
            'total_likes': total_likes,
            'total_views': total_views,
            'total_comments': total_comments,
            'average_likes': round(avg_likes, 2),
            'average_views': round(avg_views, 2),
            'average_comments': round(avg_comments, 2),
            'overall_engagement_rate': round(((total_likes + total_comments) / total_views) * 100, 2) if total_views > 0 else 0,
            'scraped_at': datetime.now().isoformat(),
            'logged_in': bool(username)
        }
    else:
        summary = {
            'creator_name': creator_name,
            'total_reels_analyzed': len(reel_urls),
            'successful_scrapes': 0,
            'failed_scrapes': len(reel_urls),
            'error': 'No successful scrapes',
            'scraped_at': datetime.now().isoformat(),
            'logged_in': bool(username)
        }

    # Save detailed analytics JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analytics_file = output_folder / f'{creator_name}_analytics_{timestamp}.json'
    output_data = {
        'summary': summary,
        'reels': analytics_data
    }

    with open(analytics_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"ANALYTICS SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully analyzed: {success_count}/{len(reel_urls)}")
    print(f"Failed: {len(reel_urls) - success_count}")

    if successful_analytics:
        print(f"\nOverall Statistics:")
        print(f"   Total Likes: {summary['total_likes']:,}")
        print(f"   Total Views: {summary['total_views']:,}")
        print(f"   Total Comments: {summary['total_comments']:,}")
        print(f"   Average Likes per Reel: {summary['average_likes']:,.2f}")
        print(f"   Average Views per Reel: {summary['average_views']:,.2f}")
        print(f"   Average Comments per Reel: {summary['average_comments']:,.2f}")
        print(f"   Overall Engagement Rate: {summary['overall_engagement_rate']}%")

    print(f"\nSaved to:")
    print(f"   • {analytics_file}")
    print(f"{'='*70}\n")

    return analytics_data


# ============================================
# MAIN SCRIPT
# ============================================

if __name__ == "__main__":

    # All reel URLs (shortened list for example; keep your full list here)
    reel_urls = [
        "https://www.instagram.com/runtimebrt/reel/DMaeWnlvG4p/",
        "https://www.instagram.com/runtimebrt/reel/DMelIKWPHVW/",
        "https://www.instagram.com/runtimebrt/reel/DMkLREsvjKe/",
        "https://www.instagram.com/runtimebrt/reel/DMnYh4RPlou/",
        "https://www.instagram.com/runtimebrt/reel/DMxaytfBdVD/",
        "https://www.instagram.com/runtimebrt/reel/DM0OgvlP8tS/",
        "https://www.instagram.com/runtimebrt/reel/DM2hcxGv2j1/",
        "https://www.instagram.com/runtimebrt/reel/DM5FvWUP4fF/",
        "https://www.instagram.com/runtimebrt/reel/DM96RswPCeF/",
        "https://www.instagram.com/runtimebrt/reel/DNAVMQzPpp5/",
        "https://www.instagram.com/runtimebrt/reel/DNDF59dP22v/",
        "https://www.instagram.com/runtimebrt/reel/DNF_58TP60y/",
        "https://www.instagram.com/runtimebrt/reel/DNIHf8IvOH_/",
        "https://www.instagram.com/caleb_friesen/reel/DNNUwB1T1VU/",
        "https://www.instagram.com/caleb_friesen/reel/DNP4t1pzCJM/",
        "https://www.instagram.com/runtimebrt/reel/DNTba0Jv_tg/",
        "https://www.instagram.com/runtimebrt/reel/DNUrB_MvlYY/",
        "https://www.instagram.com/runtimebrt/reel/DNVpO_LPLCs/",
        "https://www.instagram.com/runtimebrt/reel/DNW9cU-v3mt/",
        "https://www.instagram.com/runtimebrt/reel/DNbAPOcv-KD/",
        "https://www.instagram.com/runtimebrt/reel/DNgN7TfPAqQ/",
        "https://www.instagram.com/runtimebrt/reel/DNivIAaPpRY/",
        "https://www.instagram.com/runtimebrt/reel/DNlHE-VPNqu/",
        "https://www.instagram.com/runtimebrt/reel/DNnX-I-vVXF/",
        "https://www.instagram.com/runtimebrt/reel/DNnvs4rvdIC/",
        "https://www.instagram.com/runtimebrt/reel/DNqPsdVvxYK/",
        "https://www.instagram.com/runtimebrt/reel/DNrvQQ03vvM/",
        "https://www.instagram.com/runtimebrt/reel/DNx1A_oXmcC/",
        "https://www.instagram.com/runtimebrt/reel/DNx8P7LXhHd/",
        "https://www.instagram.com/runtimebrt/reel/DNzxd603gjj/",
        "https://www.instagram.com/runtimebrt/reel/DN0YQX5UAAB/",
        "https://www.instagram.com/runtimebrt/reel/DN0d4mYUDlR/",
        "https://www.instagram.com/runtimebrt/reel/DN3LqBLUDsv/",
        "https://www.instagram.com/runtimebrt/reel/DN4qF-5iMDQ/",
        "https://www.instagram.com/runtimebrt/reel/DN5h5KrD2mO/",
        "https://www.instagram.com/runtimebrt/reel/DN5sjUQj_Gw/",
        "https://www.instagram.com/runtimebrt/reel/DN7uQEEiKDn/",
        "https://www.instagram.com/runtimebrt/reel/DN8HuefCKqc/",
        "https://www.instagram.com/runtimebrt/reel/DN9n9dMCClB/",
        "https://www.instagram.com/runtimebrt/reel/DN-esOjiCBC/",
        "https://www.instagram.com/runtimebrt/reel/DOCsH0pD4M2/",
        "https://www.instagram.com/runtimebrt/reel/DODib3ojyeB/",
        "https://www.instagram.com/runtimebrt/reel/DOFe1AMD349/",
        "https://www.instagram.com/runtimebrt/reel/DOIFFK2D8Fv/",
        "https://www.instagram.com/runtimebrt/reel/DOJYWpmiAhi/",
        "https://www.instagram.com/runtimebrt/reel/DOKlXyVj7jZ/",
        "https://www.instagram.com/runtimebrt/reel/DOLDx0Ij-uQ/",
        "https://www.instagram.com/runtimebrt/reel/DOMBDY1j9Qu/",
        "https://www.instagram.com/runtimebrt/reel/DONOXphj17e/",
        "https://www.instagram.com/runtimebrt/reel/DON2bXrjxwF/",
        "https://www.instagram.com/runtimebrt/reel/DOQEuuFD2HA/",
        "https://www.instagram.com/runtimebrt/reel/DOQ6WBfCDsK/",
        "https://www.instagram.com/runtimebrt/reel/DOWQRVVCDUd/",
        "https://www.instagram.com/runtimebrt/reel/DOX1qvHj59r/",
        "https://www.instagram.com/runtimebrt/reel/DOaL4M1iFN7/",
        "https://www.instagram.com/runtimebrt/reel/DObZigwDBmg/",
        "https://www.instagram.com/runtimebrt/reel/DOcrW2RiMC5/",
        "https://www.instagram.com/runtimebrt/reel/DOfcFLkCAPZ/",
        "https://www.instagram.com/runtimebrt/reel/DOi8b80jy76/",
        "https://www.instagram.com/runtimebrt/reel/DOlnruljZ_2/",
        "https://www.instagram.com/runtimebrt/reel/DOp-0IWDypL/",
        "https://www.instagram.com/runtimebrt/reel/DOsOfljDwrF/",
        "https://www.instagram.com/runtimebrt/reel/DOssjlrj6pk/",
        "https://www.instagram.com/runtimebrt/reel/DOu_c_1j3rH/",
        "https://www.instagram.com/runtimebrt/reel/DOvXTKqj8p9/",
        "https://www.instagram.com/runtimebrt/reel/DOxgOvQjx9d/",
        "https://www.instagram.com/runtimebrt/reel/DO0bUsEj-pm/",
        "https://www.instagram.com/runtimebrt/reel/DO1CnySj1tm/",
        "https://www.instagram.com/runtimebrt/reel/DO7xqYLDwZS/",
        "https://www.instagram.com/runtimebrt/reel/DO72ukFD5qL/",
        "https://www.instagram.com/runtimebrt/reel/DO-fA17Dzel/",
        "https://www.instagram.com/runtimebrt/reel/DO_D870iHB9/",
        "https://www.instagram.com/runtimebrt/reel/DPA7WfMj0P3/",
        "https://www.instagram.com/runtimebrt/reel/DPB06NYj5fm/",
        "https://www.instagram.com/runtimebrt/reel/DPDgQ-vDyXB/",
        "https://www.instagram.com/runtimebrt/reel/DPD2MHLD9DK/",
        "https://www.instagram.com/runtimebrt/reel/DPF8Bi7D8mx/",
        "https://www.instagram.com/runtimebrt/reel/DPGysobj7aK/",
        "https://www.instagram.com/runtimebrt/reel/DPN3CJgj_TD/",
        "https://www.instagram.com/runtimebrt/reel/DPOS_WvD3Sp/",
        "https://www.instagram.com/runtimebrt/reel/DPOsw4Yj2NN/",
        "https://www.instagram.com/runtimebrt/reel/DPQe1fmjyia/",
        "https://www.instagram.com/runtimebrt/reel/DPQ0u7Wjxbk/",
        "https://www.instagram.com/runtimebrt/reel/DPRKs9VDxZU/",
        "https://www.instagram.com/runtimebrt/reel/DPS1lWBD0nu/",
        "https://www.instagram.com/runtimebrt/reel/DPTo9MGD4Wy/",
        "https://www.instagram.com/runtimebrt/reel/DPVXp9QD-L6/",
        "https://www.instagram.com/runtimebrt/reel/DPVjeoOjywl/",
    ]

    creator_name = "runtimebrt"

    print("\n" + "="*70)
    print("Instagram Reel Analytics Scraper")
    print("Using Instaloader library")
    print("="*70)

    # Hardcoded placeholder credentials - edit these values
    INSTAGRAM_USERNAME = "laxm.ankumar3735"
    INSTAGRAM_PASSWORD = "laxman12345"

    print(f"\nLogin credentials are hardcoded in the script. Using username: {INSTAGRAM_USERNAME}")

    print("\n" + "="*70)
    print(f"Estimated time: {len(reel_urls) * 5 // 60} minutes")
    print("="*70)

    try:
        results = get_reel_analytics(reel_urls, creator_name, INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        print("Partial results may have been saved to the analytics folder")
