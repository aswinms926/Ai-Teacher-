# 🚀 Quick Start - Set Your Gemini API Key

## Option 1: PowerShell (Quick & Easy) ⚡

Just run this command in PowerShell:

```powershell
$env:GEMINI_API_KEY="paste-your-actual-key-here"
```

**Example:**
```powershell
$env:GEMINI_API_KEY="AIzaSyABC123def456GHI789jkl012MNO345pqr"
```

✅ **Done!** You can now run the code.

---

## Option 2: Create .env File (Permanent) 📁

### Step 1: Create the .env file

Copy the example file:
```powershell
Copy-Item .env.example .env
```

Or manually create a file named `.env` in `d:\mod\ai_tutor\`

### Step 2: Edit the .env file

Open `.env` and replace the placeholder:

```bash
# Change this line:
GEMINI_API_KEY=your-gemini-api-key-here

# To your actual key:
GEMINI_API_KEY=AIzaSyABC123def456GHI789jkl012MNO345pqr
```

### Step 3: Install python-dotenv

```powershell
pip install python-dotenv
```

### Step 4: Load it in your code

Add this at the top of your Python files:

```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

# Now your code can access the API key
from vector_store.embeddings import EmbeddingGenerator
generator = EmbeddingGenerator(provider="gemini")
```

---

## Where to Get Your Gemini API Key? 🔑

1. Go to: **https://makersuite.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (starts with `AIza...`)

---

## Test It! 🧪

After setting the key, test it:

```powershell
python test_embeddings.py
```

You should see:
```
✓ Successfully initialized GEMINI provider
✓ Embedding generated successfully
  Dimensions: 768
```

---

## Current Setup ✅

- **Default Provider**: Gemini (configured in `config.py`)
- **Model**: text-embedding-004
- **Dimensions**: 768
- **Cost**: FREE (with usage limits)

---

## Need Help?

See `GEMINI_SETUP.md` for detailed instructions and troubleshooting.
