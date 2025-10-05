// src/containers/Main.js
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
import {splashScreen} from "../portfolio";
import {StyleProvider} from "../contexts/StyleContext";
import {useLocalStorage} from "../hooks/useLocalStorage";
import "./Main.scss";
import React, {useEffect, useState, useCallback} from "react";

import Project1 from "../pages/projects/Project1";
import Project2 from "../pages/projects/Project2";
import Project3 from "../pages/projects/Project3";
import Project4 from "../pages/projects/Project4";
import GmmEmAssignmentPage from "../pages/projects/GmmEmAssignmentPage";

const Main = () => {
  const darkPref = window.matchMedia("(prefers-color-scheme: dark)");
  const [isDark, setIsDark] = useLocalStorage("isDark", darkPref.matches);
  const [isShowingSplashAnimation, setIsShowingSplashAnimation] = useState(true);
  const [currentPage, setCurrentPage] = useState("main");

  // ✅ 해시(#)가 남은 상태로 새로고침하면 강제로 "/"로 리다이렉트
  useEffect(() => {
    if (window.location.hash && window.location.pathname === "/") {
      window.history.replaceState(null, "", "/");
    }
  }, []);

  // ✅ 여기: 이동해야 할 섹션 해시를 임시로 저장
  const [pendingHash, setPendingHash] = useState(null); // e.g. "skills"

  // 원하는 해시로 스크롤 (렌더 완료까지 재시도)
  const scrollTo = useCallback((hash, retries = 20) => {
    if (!hash) return;
    const selector = hash.startsWith("#") ? hash : `#${hash}`;
    const el = document.querySelector(selector);
    if (el) {
      el.scrollIntoView({behavior: "smooth", block: "start"});
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

      if (
        path === "project1" ||
        path === "project2" ||
        path === "project3" ||
        path === "project4" ||
        path === "gmm-em-assignment"
      ) {
        setCurrentPage(path);
        setPendingHash(null);
      } else {
        setCurrentPage("main");
        setPendingHash(hash); // 메인일 때는 해시 기억 후 스크롤은 아래 useEffect에서
      }
    };

    window.addEventListener("popstate", handlePopState);

    // 초기 진입
    const initialPath = window.location.pathname.replace("/", "");
    const initialHash = window.location.hash.replace("#", "") || null;
    if (
      initialPath === "project1" ||
      initialPath === "project2" ||
      initialPath === "project3" ||
      initialPath === "project4" ||
      initialPath === "gmm-em-assignment"
    ) {
      setCurrentPage(initialPath);
    } else {
      setCurrentPage("main");
      setPendingHash(initialHash);
    }

    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // ✅ 메인으로 전환된 “뒤”에 대상을 찾아 스크롤
  useEffect(() => {
    if (currentPage === "main" && pendingHash) {
      // 다음 페인트 이후 시도
      requestAnimationFrame(() => scrollTo(pendingHash));
    }
  }, [currentPage, pendingHash, scrollTo]);

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

  // ✅ 해시 지원하도록 확장
  const changePage = (pageName, hash) => {
    setCurrentPage(pageName);
    setPendingHash(pageName === "main" ? hash || null : null);
    const url =
      pageName === "main"
        ? `/${hash ? `#${hash}` : ""}`
        : `/${pageName}${hash ? `#${hash}` : ""}`;
    window.history.pushState(null, "", url);
    // 메인일 땐 렌더 이후 useEffect가 스크롤 담당
  };

  // 페이지 바뀌면 맨 위로 (섹션 스크롤은 별도 처리)
  useEffect(() => {
    if (!pendingHash) window.scrollTo(0, 0);
  }, [currentPage, pendingHash]);

  return (
    <div className={isDark ? "dark-mode" : null}>
      <StyleProvider value={{isDark: isDark, changeTheme: changeTheme}}>
        {isShowingSplashAnimation && splashScreen.enabled ? (
          <SplashScreen />
        ) : (
          <>
            {/* ✅ changePage를 Header로 전달해야 함 */}
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
