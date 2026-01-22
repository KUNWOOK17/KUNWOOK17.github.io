---
layout: default
permalink: /
---

<div class="home-hero">
  <h1 class="home-hero__title">{{ site.title }}</h1>
  <p class="home-hero__subtitle">{{ site.description }}</p>
  <p class="home-hero__desc">
    안녕하세요! 저만의 개발 노하우와 기술적 인사이트를 공유하는 공간입니다. 🚀
  </p>
</div>

<div class="home-layout">
  <!-- 왼쪽: 최근 포스트 -->
  <section class="home-main">
    <div class="home-section-head">
      <h2 class="home-section-title">최근 포스트</h2>
      <a class="home-section-link" href="/archive/">모든 포스트 보기 →</a>
    </div>

    <div class="home-post-list">
      {% for post in site.posts limit:6 %}
      <article class="home-post-item">
        <h3 class="home-post-title">
          <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        </h3>

        <div class="home-post-meta">
          <span>{{ post.date | date: "%Y년 %m월 %d일" }}</span>
          {% if post.categories and post.categories.size > 0 %}
            · <a href="/categories/">{{ post.categories[0] }}</a>
          {% endif %}
          {% if post.author %}
            · <span>{{ post.author }}</span>
          {% endif %}
        </div>

        {% if post.excerpt %}
          <p class="home-post-excerpt">{{ post.excerpt | strip_html | truncate: 140 }}</p>
        {% endif %}

        {% if post.tags and post.tags.size > 0 %}
        <div class="home-post-tags">
          {% for tag in post.tags limit:3 %}
            <a class="home-tag" href="/tags/">#{{ tag }}</a>
          {% endfor %}
        </div>
        {% endif %}

        <div class="home-post-divider"></div>
      </article>
      {% endfor %}
    </div>
  </section>

  <!-- 오른쪽: 사이드바 -->
  <aside class="home-side">
    <!-- About 카드 -->
    <div class="side-card">
      <div class="side-card__head">
        <h3>About</h3>
        <div class="side-card__line"></div>
      </div>

      <div class="side-about">
        <!-- 이미지 경로는 너가 쓰는 아바타로 바꿔줘 -->
        <img class="side-about__avatar" src="/assets/images/logo.png" alt="profile" />
        <p class="side-about__text">
          실무 중심의 개발 이야기를 기록합니다. 문제 해결 과정과 배운 점들을 공유합니다.
        </p>
        <a class="side-about__link" href="/about/">더 알아보기 →</a>
      </div>
    </div>

    <!-- 카테고리 카드 -->
    <div class="side-card">
      <div class="side-card__head">
        <h3>카테고리</h3>
        <div class="side-card__line"></div>
      </div>

      <ul class="side-list">
        {% assign cats = site.categories | sort %}
        {% for cat in cats %}
          {% assign cat_name = cat[0] %}
          {% assign cat_posts = cat[1] %}
          <li>
            <a href="/categories/">{{ cat_name }}</a>
            <span class="side-count">({{ cat_posts | size }})</span>
          </li>
        {% endfor %}
      </ul>
      <a class="side-more" href="/categories/">모든 카테고리 보기 →</a>
    </div>
  </aside>
</div>
