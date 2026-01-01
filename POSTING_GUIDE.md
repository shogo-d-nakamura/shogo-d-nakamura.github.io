# Blog Posting Guide

A beginner's guide for creating and managing posts on your Jekyll blog.

## Quick Start

### 1. Start the Local Server

```bash
zsh viewer.sh
```

This opens your blog at `http://localhost:4000` with live reload - changes appear automatically.

### 2. Create a New Post

```bash
zsh post.sh
```

This creates a new post file in `_posts/` with today's date. Edit the file name "My New Post" in `post.sh` before running, or rename the file after creation.

**Example:** Running the script creates `_posts/2026-01-01-my-new-post.md`

## Post Structure

### Front Matter

Every post starts with YAML front matter between `---` markers:

```yaml
---
layout: post
title: Your Post Title
date: 2026-01-01 12:00 +0900
description: 'Brief description for SEO'
category: 'Category Name'
tags: [tag1, tag2]
published: true
math: true
---
```

| Field | Description |
|-------|-------------|
| `title` | Post title (displayed on page) |
| `date` | Publication date and timezone |
| `description` | SEO description (appears in search results) |
| `category` | Single category (e.g., 'Machine Learning', 'Chemistry') |
| `tags` | List of tags in brackets |
| `published` | Set to `false` to hide the post |
| `math` | Enables LaTeX equations (enabled by default) |

### Writing Content

After the front matter, write your content in Markdown:

```markdown
# Heading 1
## Heading 2
### Heading 3

Regular paragraph text.

**Bold text** and *italic text*.

- Bullet list item
- Another item

1. Numbered list
2. Second item

[Link text](https://example.com)

![Image alt text](/assets/img/your-image.png)

`inline code`

​```python
# Code block with syntax highlighting
def hello():
    print("Hello, World!")
​```
```

### Math Equations

Your blog supports LaTeX math equations:

**Inline math:** Use single dollar signs
```
The probability $p(x)$ is calculated as...
```

**Display math:** Use double dollar signs
```
$$
\hat{\mu} = \frac{1}{N}\sum_{n=1}^{N}x^{(n)}
$$
```

## Bilingual Posts (Japanese/English)

Your blog supports both Japanese and English versions of posts.

### Japanese Post (Default)

Create normally - Japanese is the default:
```
_posts/2026-01-01-my-post.md
```

### English Post

Add `_en` suffix before `.md`:
```
_posts/2026-01-01-my-post_en.md
```

Add `lang: en` to the front matter:
```yaml
---
layout: post
title: My Post (English Version)
lang: en
...
---
```

### Linking Japanese and English Versions

For the language toggle to work, both posts should have:
- Same base filename (before `_en`)
- Same publication date

Example pair:
- `_posts/2026-01-01-deep-learning-notes.md` (Japanese)
- `_posts/2026-01-01-deep-learning-notes_en.md` (English)

## Adding Images

1. Place images in `assets/img/` folder
2. Reference in your post:

```markdown
![Description](/assets/img/your-image.png)
```

For post-specific images, create a subfolder:
```
assets/img/posts/2026-01-01-post-name/
```

## Categories and Tags

### Categories

Use a single category per post. Existing categories in your blog:
- Machine Learning
- Chemistry
- (Add more as needed)

### Tags

Use multiple tags as an array:
```yaml
tags: [deep-learning, python, tutorial]
```

## Draft Posts

### Method 1: Set published to false

```yaml
published: false
```

The post won't appear on the live site but exists in `_posts/`.

### Method 2: Use _drafts folder

Create posts in `_drafts/` folder (without date prefix):
```
_drafts/my-draft-post.md
```

To preview drafts locally:
```bash
bundle exec jekyll s --drafts --livereload
```

## Common Tasks

### Rename a Post

1. Change the filename in `_posts/`
2. Update the `title` in front matter
3. The URL is based on the filename, not the title

### Change Post Date

1. Rename the file with new date prefix
2. Update the `date` in front matter

### Hide a Post Temporarily

Set `published: false` in front matter.

### Add Table of Contents

TOC is enabled by default. To disable for a specific post:
```yaml
toc: false
```

## Workflow Summary

1. **Start server:** `zsh viewer.sh`
2. **Create post:** `zsh post.sh` (edit script for custom title)
3. **Write content** in the created markdown file
4. **Preview** at `http://localhost:4000`
5. **Create English version** (optional): Copy file, add `_en` suffix, translate
6. **Commit and push** when ready to publish

## Troubleshooting

### Math equations not rendering

Ensure `math: true` is in your front matter (should be added by default).

### Post not appearing

- Check `published: true` in front matter
- Verify the date is not in the future
- Restart the Jekyll server

### Changes not showing

- Wait a few seconds for live reload
- Hard refresh browser (Cmd+Shift+R)
- Restart server: Stop with Ctrl+C, run `zsh viewer.sh` again
