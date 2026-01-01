# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based personal blog using the [Chirpy theme](https://github.com/cotes2020/jekyll-theme-chirpy) (v7.2+). The site is hosted on GitHub Pages at https://shogo-d-nakamura.github.io.

## Common Commands

```bash
# Start local development server with live reload (opens browser automatically)
zsh viewer.sh

# Create a new post
bundle exec jekyll compose "Post Title"
# or use the shortcut:
zsh post.sh

# Install dependencies
bundle install

# Build the site
bundle exec jekyll build

# Serve without the viewer script
bundle exec jekyll s --livereload
```

## Architecture

### Bilingual Post System

Posts support Japanese (default) and English versions with a custom language toggle feature:

- **Japanese posts**: `_posts/YYYY-MM-DD-post-name.md`
- **English posts**: `_posts/YYYY-MM-DD-post-name_en.md` (note the `_en` suffix)

The language toggle is implemented in:
- `assets/js/lang-toggle.js` - Toggle logic, URL handling, and localStorage persistence
- `assets/css/lang-toggle.css` - Animations and styling
- `_includes/topbar.html` - Language toggle buttons (JA/EN)
- `_includes/sidebar.html` - Recent posts filtering
- `_includes/update-list.html` - Post list with `data-lang` attributes

Posts should include `lang: ja` or `lang: en` in front matter. The default scope sets Japanese for `_posts/` and English for `_posts/*_en.md` files.

### Theme Customizations

This repo contains overridden theme files from the Chirpy gem:
- `_layouts/` - Custom layout modifications (archives, home, post, etc.)
- `_includes/` - Custom partials (sidebar, topbar, update-list, etc.)
- `_data/` - Locale and other data files

To find the original theme files:
```bash
bundle info --path jekyll-theme-chirpy
```

### Comments

Uses Utterances for comments (GitHub Issues-based), configured in `_config.yml`.

### Key Configuration

- Timezone: Asia/Tokyo
- Default language: Japanese (ja)
- Comments: Utterances
- Analytics: Google Analytics (G-Z691KZMBF9)
