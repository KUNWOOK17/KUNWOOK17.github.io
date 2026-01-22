---
layout: page
title: 태그
permalink: /tags/
---

{% assign tags = site.tags | sort %}
{% for tag in tags %}
  {% assign name = tag[0] %}
  {% assign posts = tag[1] %}
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
