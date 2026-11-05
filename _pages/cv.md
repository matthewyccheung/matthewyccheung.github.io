---
layout: page
permalink: /CV/
title: CV
nav: true
nav_order: 2
---

<div class="cv-links">
  <a class="btn z-depth-0" href="https://docs.google.com/document/d/1_yqqvlCgOm4hIor0TIqJXpp9NkVSvK7eH-Bk8P9KQoY/edit" target="_blank" rel="noopener noreferrer" role="button">
    Resume
  </a>
  <a class="btn z-depth-0" href="https://docs.google.com/document/d/1_XNNnHBikt8Y08k8aGd1F-zMO6rTwP6r4yf3PtKDxkQ/edit" target="_blank" rel="noopener noreferrer" role="button">
    Full CV
  </a>
</div>

{% assign cv_summary = site.data.cv.cv.sections.Summary | first %}
<p class="cv-intro">{{ cv_summary }}</p>

{% include cv/timeline.liquid %}