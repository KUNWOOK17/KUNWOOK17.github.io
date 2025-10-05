// src/containers/Main.js
import React, {useEffect, useState} from "react";
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

// 프로젝트 상세 페이지
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

  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname.replace("/", "");
      if (
        path === "project1" ||
        path === "project2" ||
        path === "project3" ||
        path === "project4" ||
        path === "gmm-em-assignment"
      ) {
        setCurrentPage(path);
      } else {
        setCurrentPage("main");
      }
    };

    window.addEventListener("popstate", handlePopState);

    const initialPath = window.location.pathname.replace("/", "");
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
    }

    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, []);

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

  const changePage = pageName => {
    setCurrentPage(pageName);
    window.history.pushState(null, "", pageName === "main" ? "/" : `/${pageName}`);
  };

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [currentPage]);

  return (
    <div className={isDark ? "dark-mode" : null}>
      <StyleProvider value={{isDark: isDark, changeTheme: changeTheme}}>
        {isShowingSplashAnimation && splashScreen.enabled ? (
          <SplashScreen />
        ) : (
          <>
            <Header />

            {/* ✅ 전체 레이아웃 최대 폭 제한 래퍼 (100% 줌에서도 70% 때 밸런스 유지) */}
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
