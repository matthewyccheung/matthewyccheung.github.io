---
layout: page
permalink: /publications/
title: publications
description: Selected publications and research highlights.
nav: true
nav_order: 3
---

<div class="publications">
  <h2 class="bibliography"><span>Research Highlights</span></h2>
  <ol class="bibliography">
    {% for pub in site.data.citations.featured_publications %}
      <li>
        <div class="row">
          <div class="col col-sm-2 abbr">
            <abbr class="badge rounded w-100">{{ pub.venue }}</abbr>
            {% if pub.image %}
              {% include figure.liquid loading="lazy" path=pub.image class="preview z-depth-1 rounded" alt=pub.image_alt %}
            {% endif %}
          </div>
          <div class="col-sm-8">
            <div class="title">{{ pub.title }}</div>
            <div class="periodical"><em>{{ pub.venue }}, {{ pub.year }}</em></div>
            <div class="links">
              {% for link in pub.links %}
                <a href="{{ link.url }}" class="btn btn-sm z-depth-0" role="button">{{ link.label }}</a>
              {% endfor %}
            </div>
            <div class="abstract">{{ pub.description }}</div>
          </div>
        </div>
      </li>
    {% endfor %}
  </ol>

  <h2 class="bibliography"><span>Other Publications</span></h2>
  <ol class="bibliography">
    {% for pub in site.data.citations.other_publications %}
      <li>
        <div class="row">
          <div class="col col-sm-2 abbr">
            <abbr class="badge rounded w-100">{{ pub.venue }}</abbr>
            {% if pub.image %}
              {% include figure.liquid loading="lazy" path=pub.image class="preview z-depth-1 rounded" alt=pub.image_alt %}
            {% endif %}
          </div>
          <div class="col-sm-8">
            <div class="title">{{ pub.title }}</div>
            <div class="periodical"><em>{{ pub.venue }}, {{ pub.year }}</em></div>
            <div class="links">
              {% for link in pub.links %}
                <a href="{{ link.url }}" class="btn btn-sm z-depth-0" role="button">{{ link.label }}</a>
              {% endfor %}
            </div>
            <div class="abstract">{{ pub.description }}</div>
          </div>
        </div>
      </li>
    {% endfor %}
  </ol>

  <h2 class="bibliography"><span>GitHub Code</span></h2>
  <ol class="bibliography">
    {% for item in site.data.citations.github_code %}
      <li>
        <div class="row">
          <div class="col-sm-10">
            <div class="title">{{ item.title }}</div>
            <div class="links">
              <a href="{{ item.url }}" class="btn btn-sm z-depth-0" role="button">GitHub</a>
            </div>
            {% if item.description %}
              <div class="abstract">{{ item.description }}</div>
            {% endif %}
          </div>
        </div>
      </li>
    {% endfor %}
  </ol>
</div>
