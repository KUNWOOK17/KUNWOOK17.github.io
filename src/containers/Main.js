// src/containers/Main.js
import React, { useEffect, useState, useCallback } from "react";
import Header from "../components/header/Header";
import Greeting from "./greeting/Greeting";
import Skills from "./skills/Skills";
import StackProgress from "./skillProgress/skillProgress";
import WorkExperience from "./workExperience/WorkExperience";
import Projects from "./projects/Projects";
import StartupProject from "./StartupProjects/StartupProject";
import Achievement from "./achievement/Achievement";
import Blogs from "./blogs/Blogs";
import Footer from "../components/footer/Footer";
import Talks from "./talks/Talks";
import Podcast from "./podcast/Podcast";
import Education from "./education/Education";
import ScrollToTopButton from "./topbutton/Top";
import Twitter from "./twitter-embed/twitter";
import Profile from "./profile/Profile";
import SplashScreen from "./splashScreen/SplashScreen";
import { splashScreen } from "../portfolio";
import { StyleProvider } from "../contexts/StyleContext";
import { useLocalStorage } from "../hooks/useLocalStorage";
import "./Main.scss";

import Project1 from "../pages/projects/Project1";
import Project2 from "../pages/projects/Project2";
import Project3 from "../pages/projects/Project3";
import Project4 from "../pages/projects/Project4";
import GmmEmAssignmentPage from "../pages/projects/GmmEmAssignmentPage";

const PROJECT_PAGES = ["project1", "project2", "project3", "project4", "gmm-em-assignment"];

const Main = () => {
  const darkPref = window.matchMedia("(prefers-color-scheme: dark)");
  const [isDark, setIsDark] = useLocalStorage("isDark", darkPref.matches);
  const [isShowingSplashAnimation, setIsShowingSplashAnimation] = useState(true);
  const [currentPage, setCurrentPage] = useState("main");

  // 루트(/)에서 해시가 남은 상태로 새로고침되면 주소를 / 로 정리
  useEffect(() => {
    if (window.location.hash && window.location.pathname === "/") {
      window.history.replaceState(null, "", "/");
    }
  }, []);

  // 섹션 이동을 위한 대기 해시
  const [pendingHash, setPendingHash] = useState(null); // e.g. "skills"

  // 프로젝트 페이지로 전환 시 짧게 노출할 스플래시
  const [isRouteSplash, setIsRouteSplash] = useState(false);
  const maybeShowRouteSplash = (pageName) => {
    const isProject = PROJECT_PAGES.includes(pageName);
    if (!isProject) return;
    setIsRouteSplash(true);
    const duration = (splashScreen && splashScreen.duration) || 1000;
    setTimeout(() => setIsRouteSplash(false), duration);
  };

  // 원하는 해시로 스크롤 (렌더 완료까지 재시도)
  const scrollTo = useCallback((hash, retries = 20) => {
    if (!hash) return;
    const selector = hash.startsWith("#") ? hash : `#${hash}`;
    const el = document.querySelector(selector);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      return true;
    }
    if (retries > 0) {
      setTimeout(() => scrollTo(hash, retries - 1), 50); // 최대 약 1초 재시도
    }
    return false;
  }, []);

  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname.replace("/", "");
      const hash = window.location.hash.replace("#", "") || null;

      if (PROJECT_PAGES.includes(path)) {
        setCurrentPage(path);
        setPendingHash(null);
      } else {
        setCurrentPage("main");
        setPendingHash(hash); // 메인일 때는 해시 기억 → 아래 useEffect에서 스크롤
      }
    };

    window.addEventListener("popstate", handlePopState);

    // 초기 진입
    const initialPath = window.location.pathname.replace("/", "");
    const initialHash = window.location.hash.replace("#", "") || null;
    if (PROJECT_PAGES.includes(initialPath)) {
      setCurrentPage(initialPath);
    } else {
      setCurrentPage("main");
      setPendingHash(initialHash);
    }

    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // 메인으로 전환된 “뒤”에 대상을 찾아 스크롤
  useEffect(() => {
    if (currentPage === "main" && pendingHash) {
      requestAnimationFrame(() => scrollTo(pendingHash));
    }
  }, [currentPage, pendingHash, scrollTo]);

  // 초기 스플래시 타이머
  useEffect(() => {
    if (splashScreen.enabled) {
      const splashTimer = setTimeout(
        () => setIsShowingSplashAnimation(false),
        splashScreen.duration
      );
      return () => clearTimeout(splashTimer);
    }
  }, []);

  const changeTheme = () => setIsDark(!isDark);

  // 페이지 변경 (해시 지원 + 전환 스플래시)
  const changePage = (pageName, hash) => {
    // 프로젝트 상세로 갈 때 전환 스플래시
    maybeShowRouteSplash(pageName);

    setCurrentPage(pageName);
    setPendingHash(pageName === "main" ? hash || null : null);

    const url =
      pageName === "main"
        ? `/${hash ? `#${hash}` : ""}`
        : `/${pageName}${hash ? `#${hash}` : ""}`;
    window.history.pushState(null, "", url);
    // 메인 해시 스크롤은 위 useEffect가 처리
  };

  // 페이지 바뀌면 맨 위로 (섹션 스크롤은 별도 처리)
  useEffect(() => {
    if (!pendingHash) window.scrollTo(0, 0);
  }, [currentPage, pendingHash]);

  return (
    <div className={isDark ? "dark-mode" : null}>
      <StyleProvider value={{ isDark: isDark, changeTheme: changeTheme }}>
        {((isShowingSplashAnimation && splashScreen.enabled) || isRouteSplash) ? (
          <SplashScreen />
        ) : (
          <>
            <Header changePage={changePage} />

            <div className="site-container">
              {currentPage === "main" && (
                <>
                  <Greeting />
                  <Skills />
                  <StackProgress />
                  <Education />
                  <WorkExperience />
                  <Projects />
                  <StartupProject changePage={changePage} />
                  <Achievement />
                  <Blogs changePage={changePage} />
                  <Talks />
                  <Twitter />
                  <Podcast />
                  <Profile />
                </>
              )}

              {currentPage === "project1" && <Project1 changePage={changePage} />}
              {currentPage === "project2" && <Project2 changePage={changePage} />}
              {currentPage === "project3" && <Project3 changePage={changePage} />}
              {currentPage === "project4" && <Project4 changePage={changePage} />}

              {currentPage === "gmm-em-assignment" && (
                <GmmEmAssignmentPage changePage={changePage} />
              )}
            </div>

            <Footer />
            <ScrollToTopButton />
          </>
        )}
      </StyleProvider>
    </div>
  );
};

export default Main;
