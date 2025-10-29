"""
Instagram Reel Downloader - Alternative Method
Uses Instaloader library which is more reliable
"""

import instaloader
import time
import json
import os
from pathlib import Path

def download_reels_programmatic(reel_urls, creator_name):
    """
    Programmatic version - can be called from other scripts
    Same as main() but accepts parameters instead of input()
    """
    return download_reels_with_instaloader(reel_urls, creator_name)

def run_downloader(reel_urls, creator_name):
    """Wrapper to call from Agent 1"""
    return download_reels_with_instaloader(reel_urls, creator_name)


def download_reels_with_instaloader(reel_urls, creator_name):
    """
    Download Instagram reels using Instaloader library
    More reliable than manual scraping
    """
    # Create Instaloader instance
    L = instaloader.Instaloader(
        download_videos=True,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern=''
    )
    
    # Create output folder
    output_folder = Path(f"downloads/{creator_name}")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    results = []
    success_count = 0
    
    print(f"\n{'='*60}")
    print(f"Downloading {len(reel_urls)} reels from @{creator_name}")
    print(f"Using Instaloader (more reliable method)")
    print(f"{'='*60}\n")
    
    for idx, reel_url in enumerate(reel_urls, 1):
        print(f"[{idx}/{len(reel_urls)}] Processing: {reel_url}")
        
        # Extract shortcode from URL
        # URL format: https://www.instagram.com/username/reel/SHORTCODE/
        shortcode = reel_url.rstrip('/').split('/')[-1]
        
        try:
            # Download post by shortcode
            post = instaloader.Post.from_shortcode(L.context, shortcode)
            
            # Custom filename
            filename = f"{creator_name}_{idx:03d}_{shortcode}.mp4"
            filepath = output_folder / filename
            
            # Check if already exists
            if filepath.exists():
                print(f"  ✓ Already exists, skipping")
                results.append({
                    'reel_url': reel_url,
                    'shortcode': shortcode,
                    'success': True,
                    'local_path': str(filepath),
                    'skipped': True
                })
                success_count += 1
                continue
            
            print(f"  → Downloading video...")
            
            # Download the video
            L.download_post(post, target=str(output_folder))
            
            # Instaloader saves with its own naming, rename it
            # It saves as: YYYY-MM-DD_HH-MM-SS_UTC.mp4
            # Find the most recently created mp4 file
            mp4_files = sorted(output_folder.glob("*.mp4"), key=os.path.getmtime, reverse=True)
            
            if mp4_files:
                latest_file = mp4_files[0]
                # Rename to our format
                latest_file.rename(filepath)
                
                file_size = filepath.stat().st_size / (1024*1024)  # MB
                print(f"  ✓ Downloaded successfully ({file_size:.2f} MB)")
                
                results.append({
                    'reel_url': reel_url,
                    'shortcode': shortcode,
                    'success': True,
                    'local_path': str(filepath),
                    'file_size_mb': round(file_size, 2)
                })
                success_count += 1
            else:
                print(f"  ✗ Download completed but file not found")
                results.append({
                    'reel_url': reel_url,
                    'shortcode': shortcode,
                    'success': False,
                    'error': 'File not found after download'
                })
        
        except instaloader.exceptions.InstaloaderException as e:
            print(f"  ✗ Instaloader error: {e}")
            results.append({
                'reel_url': reel_url,
                'shortcode': shortcode,
                'success': False,
                'error': str(e)
            })
        
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append({
                'reel_url': reel_url,
                'shortcode': shortcode,
                'success': False,
                'error': str(e)
            })
        
        # Delay to avoid rate limiting
        if idx < len(reel_urls):
            print(f"  → Waiting 3 seconds...\n")
            time.sleep(3)
    
    # Clean up extra files that Instaloader might create
    for txt_file in output_folder.glob("*.txt"):
        txt_file.unlink()
    for json_file in output_folder.glob("*.json.xz"):
        json_file.unlink()
    
    # Save results
    results_file = output_folder / 'download_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, indent=2, fp=f)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Successfully downloaded: {success_count}/{len(reel_urls)}")
    print(f"✗ Failed: {len(reel_urls) - success_count}")
    print(f"📁 Saved to: {output_folder}/")
    print(f"📊 Results log: {results_file}")
    print(f"{'='*60}\n")
    
    return results


# ============================================
# MAIN SCRIPT 
# ============================================

if __name__ == "__main__":
    
    # All 88 reel URLs
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
    
    print("\n🚀 Instagram Reel Downloader")
    print("📦 Using Instaloader library (more reliable)")
    print("⏱️  Estimated time: 5-7 minutes\n")
    
    try:
        results = download_reels_with_instaloader(reel_urls, creator_name)
    except KeyboardInterrupt:
        print("\n\n  Download interrupted by user")
        print("Run again to resume - already downloaded files will be skipped")