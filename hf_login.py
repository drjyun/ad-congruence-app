from huggingface_hub import login
import getpass

print("=" * 60)
print("Hugging Face Login")
print("=" * 60)
print("\nTo get your token:")
print("1. Go to: https://huggingface.co/settings/tokens")
print("2. Create a new token with 'Write' access")
print("3. Copy and paste it below\n")

token = getpass.getpass("Enter your HF token (will be hidden): ")
login(token=token, add_to_git_credential=True)
print("\n✓ Successfully logged in to Hugging Face!")
print("✓ Git credentials configured")

