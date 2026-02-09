document.addEventListener("DOMContentLoaded", function() {
    // Only run if we are in the docs
    var path = window.location.pathname;
    
    // Check if we are in zh or en
    var isZh = path.indexOf('/zh/') !== -1;
    var isEn = path.indexOf('/en/') !== -1;
    
    // If not in either, do nothing (or maybe we are at root)
    if (!isZh && !isEn) return;
    
    var currentLang = isZh ? 'zh' : 'en';
    
    // Create the select container
    var container = document.createElement('div');
    container.className = 'lang-switcher-container';
    container.style.padding = '10px';
    container.style.textAlign = 'center';
    
    // Create label
    var label = document.createElement('label');
    label.textContent = 'Language: ';
    label.style.color = '#ccc';
    label.style.marginRight = '5px';
    label.style.fontSize = '0.9em';
    
    // Create select element
    var select = document.createElement('select');
    select.className = 'lang-switcher-select';
    select.style.padding = '5px';
    select.style.borderRadius = '4px';
    select.style.border = '1px solid #ccc';
    select.style.backgroundColor = '#fcfcfc';
    select.style.color = '#333';
    select.style.cursor = 'pointer';
    
    // Options
    var optionZh = document.createElement('option');
    optionZh.value = 'zh';
    optionZh.textContent = '简体中文';
    optionZh.selected = isZh;
    
    var optionEn = document.createElement('option');
    optionEn.value = 'en';
    optionEn.textContent = 'English';
    optionEn.selected = isEn;
    
    select.appendChild(optionZh);
    select.appendChild(optionEn);
    
    // Event listener
    select.addEventListener('change', function() {
        var newLang = this.value;
        if (newLang === currentLang) return;
        
        var targetUrl = path.replace('/' + currentLang + '/', '/' + newLang + '/');
        window.location.href = targetUrl;
    });
    
    container.appendChild(label);
    container.appendChild(select);
    
    // Inject into the sidebar
    var nav = document.querySelector('.wy-side-nav-search');
    if (nav) {
        // Insert after the search box or title
        nav.appendChild(container);
    } else {
        // Fallback: Fixed position
        container.style.position = 'fixed';
        container.style.bottom = '20px';
        container.style.right = '20px';
        container.style.backgroundColor = '#2980b9';
        container.style.borderRadius = '5px';
        container.style.zIndex = '9999';
        
        label.style.color = 'white';
        document.body.appendChild(container);
    }
});
