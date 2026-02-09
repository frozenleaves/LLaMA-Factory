document.addEventListener("DOMContentLoaded", function() {
    // Only run if we are in the docs
    var path = window.location.pathname;
    
    // Check if we are in zh or en
    var isZh = path.indexOf('/zh/') !== -1;
    var isEn = path.indexOf('/en/') !== -1;
    
    if (!isZh && !isEn) return;
    
    var currentLang = isZh ? 'zh' : 'en';
    var targetLang = isZh ? 'en' : 'zh';
    var targetLabel = isZh ? 'English' : '简体中文';
    
    // Create the button
    var btn = document.createElement('a');
    btn.className = 'lang-switcher-btn';
    btn.textContent = targetLabel;
    
    // Calculate target URL
    // Replace the first occurrence of /zh/ or /en/
    var targetUrl = path.replace('/' + currentLang + '/', '/' + targetLang + '/');
    btn.href = targetUrl;
    
    // Style the button
    btn.style.position = 'fixed';
    btn.style.bottom = '20px';
    btn.style.right = '20px';
    btn.style.zIndex = '9999';
    btn.style.padding = '8px 16px';
    btn.style.backgroundColor = '#2980b9'; // Sphinx blue
    btn.style.color = 'white';
    btn.style.borderRadius = '4px';
    btn.style.textDecoration = 'none';
    btn.style.fontWeight = 'bold';
    btn.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
    btn.style.cursor = 'pointer';
    
    // Add hover effect via JS since we are using inline styles
    btn.onmouseover = function() {
        this.style.backgroundColor = '#3091d1';
    };
    btn.onmouseout = function() {
        this.style.backgroundColor = '#2980b9';
    };
    
    // Add to body
    document.body.appendChild(btn);
    
    // Also try to add to the side nav if available (sphinx_rtd_theme)
    var nav = document.querySelector('.wy-side-nav-search');
    if (nav) {
        var navBtn = btn.cloneNode(true);
        // Reset fixed positioning for nav button
        navBtn.style.position = 'static';
        navBtn.style.display = 'inline-block';
        navBtn.style.marginTop = '10px';
        navBtn.style.marginBottom = '10px';
        navBtn.style.marginRight = '0';
        navBtn.style.boxShadow = 'none';
        navBtn.style.background = 'rgba(255,255,255,0.2)';
        navBtn.onmouseover = function() { this.style.background = 'rgba(255,255,255,0.3)'; };
        navBtn.onmouseout = function() { this.style.background = 'rgba(255,255,255,0.2)'; };
        
        nav.appendChild(navBtn);
        
        // If we added to nav, maybe we don't need the floating one?
        // Let's keep both or decide. Floating is safer if theme changes.
        // But for RTD theme, Nav is better. Let's hide floating if nav exists.
        btn.style.display = 'none';
    }
});
