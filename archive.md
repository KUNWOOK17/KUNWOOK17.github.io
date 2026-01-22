---
layout: page
title: 아카이브
permalink: /archive/
---

<div class="archive-timeline">
  {% assign posts_by_year = site.posts | group_by_exp: "post", "post.date | date: '%Y'" | reverse %}

  {% for year in posts_by_year %}
    <section class="archive-year">
      <div class="archive-year__head">
        <h2 class="archive-year__title">{{ year.name }}년</h2>
        <div class="archive-year__count">{{ year.items | size }}개 포스트</div>
      </div>

      <div class="archive-list">
        {% for post in year.items %}
          <article class="archive-item">
            <div class="archive-item__dot" aria-hidden="true"></div>

            <div class="archive-item__body">
              <div class="archive-item__date">
                {{ post.date | date: "%m.%d" }}
              </div>

              <a class="archive-item__title" href="{{ post.url | relative_url }}">
                {{ post.title }}
              </a>

              {% if post.excerpt %}
                <div class="archive-item__excerpt">
                  {{ post.excerpt | strip_html | truncate: 140 }}
                </div>
              {% endif %}

              <div class="archive-item__meta">
                {% if post.categories and post.categories.size > 0 %}
                  <span class="archive-item__cats">
                    {% for c in post.categories %}
                      <span class="archive-pill">{{ c }}</span>
                    {% endfor %}
                  </span>
                {% endif %}

                {% if post.tags and post.tags.size > 0 %}
                  <span class="archive-item__tags">
                    {% for t in post.tags limit: 5 %}
                      <span class="archive-pill archive-pill--tag">{{ t }}</span>
                    {% endfor %}
                  </span>
                {% endif %}
              </div>
            </div>
          </article>
        {% endfor %}
      </div>
    </section>
  {% endfor %}
</div>
