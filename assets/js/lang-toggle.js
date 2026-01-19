document.addEventListener('DOMContentLoaded', function() {
  const langJaBtn = document.getElementById('lang-ja');
  const langEnBtn = document.getElementById('lang-en');
  const postList = document.getElementById('post-list');
  
  if (!langJaBtn || !langEnBtn) return;
  
  // Cache DOM elements for performance
  const body = document.body;
  const otherPostItems = document.querySelectorAll('[data-lang]');
  
  // Get language from URL parameter, localStorage, or detect from current page
  let currentLang = getInitialLanguage();
  
  // If we're on a post page, also check the current post's language
  if (isOnPostPage()) {
    const postLang = getCurrentPostLanguage();
    
    // Only override if we don't have a URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (!urlParams.get('lang')) {
      currentLang = postLang;
    }
  }
  
  // Initialize the display
  if (isOnPostPage()) {
    // On post pages, just update button states
    updateButtonStates(currentLang);
  } else {
    // On list pages, do full display update
    updateLanguageDisplay(currentLang);
  }
  
  // Add click event listeners with improved handling
  langJaBtn.addEventListener('click', () => switchLanguage('ja'));
  langEnBtn.addEventListener('click', () => switchLanguage('en'));
  
  // Add keyboard shortcuts (Alt+L to toggle)
  document.addEventListener('keydown', function(e) {
    if (e.altKey && e.key.toLowerCase() === 'l') {
      e.preventDefault();
      const newLang = currentLang === 'ja' ? 'en' : 'ja';
      switchLanguage(newLang);
    }
  });
  
  // Handle browser back/forward buttons for URL-based routing
  window.addEventListener('popstate', function(e) {
    if (e.state && e.state.lang) {
      currentLang = e.state.lang;
      updateLanguageDisplay(currentLang, false); // Don't update URL again
    }
  });
  
  // Helper functions
  function isOnPostPage() {
    // Check if we're on a blog post page by looking for post-specific elements
    // or URL patterns
    return document.body.classList.contains('post') || 
           window.location.pathname.startsWith('/posts/') ||
           window.location.pathname.match(/\/\d{4}\/\d{2}\/\d{2}\//) ||
           document.querySelector('article.post') !== null;
  }
  
  function getCurrentPostLanguage() {
    // Try to detect current post language from various sources
    
    // 1. Check if URL contains _en pattern (English posts)
    if (window.location.pathname.includes('_en')) {
      return 'en';
    }
    
    // 2. Check HTML lang attribute
    const htmlLang = document.documentElement.lang;
    if (htmlLang === 'en' || htmlLang === 'ja') {
      return htmlLang;
    }
    
    // 3. Check for language indicators in the content
    const langMeta = document.querySelector('meta[name="page-lang"]');
    if (langMeta) {
      return langMeta.getAttribute('content');
    }
    
    // 4. Check Open Graph locale
    const ogLocale = document.querySelector('meta[property="og:locale"]');
    if (ogLocale) {
      const locale = ogLocale.getAttribute('content');
      if (locale === 'en' || locale === 'ja') {
        return locale;
      }
    }
    
    // 5. Default assumption - English posts have _en in URL, others are Japanese
    return window.location.pathname.includes('_en') ? 'en' : 'ja';
  }
  
  function getInitialLanguage() {
    // Check URL parameter first
    const urlParams = new URLSearchParams(window.location.search);
    const urlLang = urlParams.get('lang');
    if (urlLang === 'ja' || urlLang === 'en') {
      return urlLang;
    }
    
    // Fallback to localStorage or default
    try {
      return localStorage.getItem('site-language') || 'ja';
    } catch (e) {
      console.warn('localStorage not available, using default language');
      return 'ja';
    }
  }
  
  function switchLanguage(lang) {
    if (currentLang === lang) return;
    
    currentLang = lang;
    
    // Store preference safely
    try {
      localStorage.setItem('site-language', currentLang);
    } catch (e) {
      console.warn('Could not save language preference');
    }
    
    // Check if we're on a blog post page
    if (isOnPostPage()) {
      handlePostPageLanguageSwitch(lang);
    } else {
      // Update display and URL for list pages
      updateLanguageDisplay(currentLang, true);
    }
    
    // Add visual feedback
    const targetBtn = lang === 'ja' ? langJaBtn : langEnBtn;
    targetBtn.classList.add('lang-switching');
    setTimeout(() => {
      targetBtn.classList.remove('lang-switching');
    }, 300);
  }
  
  function updateButtonStates(lang) {
    // Update button states with improved accessibility
    langJaBtn.classList.toggle('active', lang === 'ja');
    langEnBtn.classList.toggle('active', lang === 'en');
    langJaBtn.setAttribute('aria-pressed', lang === 'ja');
    langEnBtn.setAttribute('aria-pressed', lang === 'en');
  }
  
  function updateLanguageDisplay(lang, updateUrl = true) {
    // Update button states
    updateButtonStates(lang);
    
    // Update body class for CSS filtering
    body.classList.remove('lang-filter-ja', 'lang-filter-en');
    body.classList.add('lang-filter-' + lang);
    
    // Update URL if requested
    if (updateUrl) {
      updateUrlParameter(lang);
    }
    
    // Show/hide posts based on language with improved performance
    if (postList) {
      const posts = postList.querySelectorAll('.post-item');
      let visibleCount = 0;
      
      posts.forEach(post => {
        const postLang = post.getAttribute('data-lang') || 'ja';
        const isVisible = postLang === lang;
        post.style.display = isVisible ? '' : 'none';
        if (isVisible) visibleCount++;
      });
      
      // Show/hide empty state message if no posts in selected language
      updateEmptyState(visibleCount === 0, lang);
    }
    
    // Update other page elements that show posts (sidebar, archives, etc.)
    updateOtherPostLists(lang);
  }
  
  function updateUrlParameter(lang) {
    const url = new URL(window.location);
    url.searchParams.set('lang', lang);
    
    // Use replaceState to avoid creating history entries for language changes
    history.replaceState({ lang: lang }, '', url);
  }
  
  function handlePostPageLanguageSwitch(targetLang) {
    // Get current page information
    const currentPath = window.location.pathname;
    const currentLang = getCurrentPostLanguage();
    
    if (currentLang === targetLang) return;
    
    // Try to find the corresponding post in the target language
    const alternativeUrl = getAlternativePostUrl(currentPath, currentLang, targetLang);
    
    if (alternativeUrl) {
      // Navigate to the corresponding post
      const finalUrl = alternativeUrl + (targetLang !== 'ja' ? '?lang=' + targetLang : '');
      window.location.href = finalUrl;
    } else {
      // If no corresponding post exists, go to home page with language filter
      const homeUrl = window.location.origin + '/' + (targetLang !== 'ja' ? '?lang=' + targetLang : '');
      window.location.href = homeUrl;
    }
  }
  
  function getAlternativePostUrl(currentPath, currentLang, targetLang) {
    // Handle Jekyll post URLs like /posts/post-name/ or /posts/post-name_en/
    
    let alternativePath;
    
    if (currentLang === 'ja' && targetLang === 'en') {
      // Switch from Japanese to English: add _en suffix
      // /posts/post-name/ -> /posts/post-name_en/
      alternativePath = currentPath.replace(/\/$/, '_en/');
    } else if (currentLang === 'en' && targetLang === 'ja') {
      // Switch from English to Japanese: remove _en suffix
      // /posts/post-name_en/ -> /posts/post-name/
      alternativePath = currentPath.replace(/_en\/$/, '/');
    } else {
      return null;
    }
    
    // Return the full URL
    return window.location.origin + alternativePath;
  }
  
  function updateEmptyState(isEmpty, lang) {
    let emptyMessage = document.getElementById('lang-empty-message');
    
    if (isEmpty) {
      if (!emptyMessage) {
        emptyMessage = document.createElement('div');
        emptyMessage.id = 'lang-empty-message';
        emptyMessage.className = 'text-center text-muted py-5';
        
        if (postList) {
          postList.appendChild(emptyMessage);
        }
      }
      
      const langName = lang === 'ja' ? '日本語' : 'English';
      emptyMessage.innerHTML = `
        <i class="fas fa-language fa-2x mb-3"></i>
        <p>No posts available in ${langName}</p>
        <small>Switch to ${lang === 'ja' ? 'English' : '日本語'} to see available content</small>
      `;
      emptyMessage.style.display = 'block';
    } else {
      if (emptyMessage) {
        emptyMessage.style.display = 'none';
      }
    }
  }
  
  function updateOtherPostLists(lang) {
    // Update sidebar recent posts with improved performance
    const sidebarPosts = document.querySelectorAll('#access-lastmod .content li');
    sidebarPosts.forEach(item => {
      // Use data-lang attribute if available, otherwise fall back to URL detection
      let postLang = item.getAttribute('data-lang');
      
      if (!postLang) {
        const link = item.querySelector('a');
        if (link) {
          const href = link.getAttribute('href');
          // Check if it's an English post (contains _en in URL path)
          const isEnglishPost = href.includes('_en/');
          postLang = isEnglishPost ? 'en' : 'ja';
        } else {
          postLang = 'ja'; // default
        }
      }
      
      item.style.display = postLang === lang ? '' : 'none';
    });
    
    // Update any other post listings on the page (using cached elements)
    otherPostItems.forEach(item => {
      if (item.classList.contains('post-item')) return; // Already handled
      
      const postLang = item.getAttribute('data-lang') || 'ja';
      item.style.display = postLang === lang ? '' : 'none';
    });
  }
  
  // Initialize on page load
  updateLanguageDisplay(currentLang);
});