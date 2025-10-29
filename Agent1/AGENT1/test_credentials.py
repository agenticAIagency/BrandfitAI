import instaloader

# Your credentials
USERNAME = "fafex62293"
PASSWORD = "dsjhdsgdygxch5#DGFGFGv"

print("Testing Instagram login...")

L = instaloader.Instaloader()

try:
    L.login(USERNAME, PASSWORD)
    print("✅ Login successful!")
    print(f"Logged in as: @{USERNAME}")
except instaloader.exceptions.BadCredentialsException:
    print("❌ Login failed - Wrong username or password")
except instaloader.exceptions.TwoFactorAuthRequiredException:
    print("❌ Two-factor authentication is enabled")
    print("   You need to disable 2FA or use a different method")
except Exception as e:
    print(f"❌ Error: {e}")