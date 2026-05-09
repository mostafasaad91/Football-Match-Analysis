# Contributing to WhoScored Post-Match Analyzer

First off, thank you for considering contributing to this project! It's people like you that make this tool better for everyone.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Match URL** that caused the issue
- **Python version** and OS
- **Error message** and full traceback
- **Steps to reproduce** the behavior

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use case** — why is this enhancement useful?
- **Expected behavior** — what should happen?
- **Current behavior** — what happens instead?

### Adding Team Colors

If your team is missing from the color database:

1. Find the team's official kit colors (home, accent/stripe, away)
2. Convert colors to hex format
3. Add an entry to `TOP5_2025_26_TEAM_PALETTES` following the existing format:
   ```python
   "Team Name": ["#HEXHOME", "#HEXACCENT", "#HEXAWAY"],
   ```
4. Add any common aliases to `TEAM_ALIASES`

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test with at least one match URL
5. Commit with a descriptive message
6. Push to your fork
7. Open a Pull Request

### Code Style

- Follow the existing code style and naming conventions
- Keep the dark-mode aesthetic for all visualizations
- Test your changes with multiple match URLs from different leagues
- Ensure no new warnings are introduced

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/whoscored-post-match-analyzer.git
cd whoscored-post-match-analyzer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

Thank you for contributing! 🎉
