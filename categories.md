---
layout: page
title: 카테고리
permalink: /categories/
---

{% assign categories = site.categories | sort %}
{% for category in categories %}
  {% assign name = category[0] %}
  {% assign posts = category[1] %}
  <h2 id="{{ name | slugify }}">{{ name }} ({{ posts | size }})</h2>
  <ul>
    {% for post in posts %}
      <li>
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        <span style="opacity:.7;">— {{ post.date | date: "%Y-%m-%d" }}</span>
      </li>
    {% endfor %}
  </ul>
{% endfor %}
