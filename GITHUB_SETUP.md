# Setting up LEIA on GitHub

Your LEIA repository has been initialized locally. Follow these steps to create a private GitHub repository and push your code.

## Step 1: Create a Private Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the details:
   - **Repository name**: `leia` (or your preferred name)
   - **Description**: "Low-rank Error-Informed Adjustment for improving classifier robustness"
   - **Visibility**: Select **Private** ✓
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these commands in the LEIA directory:

```bash
cd /home/abinitha/scratch/abinitha/afr-approaches/leia

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/leia.git

# Or if you prefer SSH (if you have SSH keys set up):
# git remote add origin git@github.com:YOUR_USERNAME/leia.git

# Push to GitHub
git push -u origin main
```

## Step 3: Verify

1. Go to your repository page on GitHub
2. You should see all your files there
3. The repository should be marked as "Private"

## Alternative: Using GitHub CLI

If you have GitHub CLI (`gh`) installed:

```bash
cd /home/abinitha/scratch/abinitha/afr-approaches/leia

# Create private repository and push in one command
gh repo create leia --private --source=. --remote=origin --push
```

## Troubleshooting

### If you get authentication errors:
- For HTTPS: You may need to use a Personal Access Token instead of your password
- For SSH: Make sure your SSH key is added to GitHub

### To check your remote:
```bash
git remote -v
```

### To update the remote URL:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/leia.git
```

## Next Steps

After pushing to GitHub, you can:
- Add collaborators (Settings → Collaborators)
- Set up GitHub Actions for CI/CD (if needed)
- Create issues and project boards
- Add a LICENSE file if you want to open-source it later
