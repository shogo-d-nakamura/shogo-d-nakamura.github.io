#!/bin/zsh

# Check if a title argument is provided
if [ -z "$1" ]; then
    echo "Usage: zsh post.sh \"Post Title\""
    exit 1
fi

TITLE="$1"

# Create the post
bundle exec jekyll compose "$TITLE"

# Find the most recently created post file
POST_FILE=$(ls -t _posts/*.md | head -1)

if [ -f "$POST_FILE" ]; then
    # Generate ref from filename (remove date prefix and .md extension)
    REF=$(basename "$POST_FILE" | sed 's/^[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}-//' | sed 's/\.md$//')

    # Add math: true, lang: ja, and ref fields before the closing ---
    sed -i '' "s/^---$/math: true\nlang: ja\nref: $REF\n---/" "$POST_FILE"

    echo "Created: $POST_FILE"
    echo "Added: math: true, lang: ja, ref: $REF"
else
    echo "Error: Could not find created post file"
    exit 1
fi
