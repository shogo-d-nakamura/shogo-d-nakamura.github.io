---
---

document.addEventListener('DOMContentLoaded', function() {
  const langJaBtn = document.getElementById('lang-ja');
  const langEnBtn = document.getElementById('lang-en');
  const postList = document.getElementById('post-list');
  
  if (!langJaBtn || !langEnBtn) return;
  
  // Get stored language preference or default to Japanese
  let currentLang = localStorage.getItem('site-language') || 'ja';
  
  // Initialize the display
  updateLanguageDisplay(currentLang);
  
  // Add click event listeners
  langJaBtn.addEventListener('click', function() {
    if (currentLang !== 'ja') {
      currentLang = 'ja';
      localStorage.setItem('site-language', currentLang);
      updateLanguageDisplay(currentLang);
      
      // Add animation class
      langJaBtn.classList.add('lang-switching');
      setTimeout(() => {
        langJaBtn.classList.remove('lang-switching');
      }, 300);
    }
  });
  
  langEnBtn.addEventListener('click', function() {
    if (currentLang !== 'en') {
      currentLang = 'en';
      localStorage.setItem('site-language', currentLang);
      updateLanguageDisplay(currentLang);
      
      // Add animation class
      langEnBtn.classList.add('lang-switching');
      setTimeout(() => {
        langEnBtn.classList.remove('lang-switching');
      }, 300);
    }
  });
  
  function updateLanguageDisplay(lang) {
    // Update button states
    langJaBtn.classList.toggle('active', lang === 'ja');
    langEnBtn.classList.toggle('active', lang === 'en');
    
    // Update body class for CSS filtering
    document.body.classList.remove('lang-filter-ja', 'lang-filter-en');
    document.body.classList.add('lang-filter-' + lang);
    
    // Show/hide posts based on language
    if (postList) {
      const posts = postList.querySelectorAll('.post-item');
      posts.forEach(post => {
        const postLang = post.getAttribute('data-lang') || 'ja';
        if (postLang === lang) {
          post.style.display = '';
        } else {
          post.style.display = 'none';
        }
      });
      
      // Check if there are any visible posts
      const visiblePosts = Array.from(posts).filter(post => 
        post.style.display !== 'none'
      );
      
      // Show/hide empty state message if no posts in selected language
      updateEmptyState(visiblePosts.length === 0, lang);
    }
    
    // Update other page elements that show posts (sidebar, archives, etc.)
    updateOtherPostLists(lang);
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
    // Update sidebar recent posts
    const sidebarPosts = document.querySelectorAll('#access-lastmod .content li');
    sidebarPosts.forEach(item => {
      const link = item.querySelector('a');
      if (link) {
        const href = link.getAttribute('href');
        // Simple check if it's an English post (contains _en in URL)
        const isEnglishPost = href.includes('_en/');
        const postLang = isEnglishPost ? 'en' : 'ja';
        
        if (postLang === lang) {
          item.style.display = '';
        } else {
          item.style.display = 'none';
        }
      }
    });
    
    // Update trending tags (if they exist)
    // This is a simplified version - you might want to enhance this
    // based on your specific requirements
    
    // Update any other post listings on the page
    const otherPostItems = document.querySelectorAll('[data-lang]');
    otherPostItems.forEach(item => {
      if (item.classList.contains('post-item')) return; // Already handled
      
      const postLang = item.getAttribute('data-lang') || 'ja';
      if (postLang === lang) {
        item.style.display = '';
      } else {
        item.style.display = 'none';
      }
    });
  }
  
  // Initialize on page load
  updateLanguageDisplay(currentLang);
});