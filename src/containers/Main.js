// // // import React, {useEffect, useState} from "react";
// // // import Header from "../components/header/Header";
// // // import Greeting from "./greeting/Greeting";
// // // import Skills from "./skills/Skills";
// // // import StackProgress from "./skillProgress/skillProgress";
// // // import WorkExperience from "./workExperience/WorkExperience";
// // // import Projects from "./projects/Projects";
// // // import StartupProject from "./StartupProjects/StartupProject";
// // // import Achievement from "./achievement/Achievement";
// // // import Blogs from "./blogs/Blogs";
// // // import Footer from "../components/footer/Footer";
// // // import Talks from "./talks/Talks";
// // // import Podcast from "./podcast/Podcast";
// // // import Education from "./education/Education";
// // // import ScrollToTopButton from "./topbutton/Top";
// // // import Twitter from "./twitter-embed/twitter";
// // // import Profile from "./profile/Profile";
// // // import SplashScreen from "./splashScreen/SplashScreen";
// // // import {splashScreen} from "../portfolio";
// // // import {StyleProvider} from "../contexts/StyleContext";
// // // import {useLocalStorage} from "../hooks/useLocalStorage";
// // // import "./Main.scss";

// // // const Main = () => {
// // //   const darkPref = window.matchMedia("(prefers-color-scheme: dark)");
// // //   const [isDark, setIsDark] = useLocalStorage("isDark", darkPref.matches);
// // //   const [isShowingSplashAnimation, setIsShowingSplashAnimation] =
// // //     useState(true);

// // //   useEffect(() => {
// // //     if (splashScreen.enabled) {
// // //       const splashTimer = setTimeout(
// // //         () => setIsShowingSplashAnimation(false),
// // //         splashScreen.duration
// // //       );
// // //       return () => {
// // //         clearTimeout(splashTimer);
// // //       };
// // //     }
// // //   }, []);

// // //   const changeTheme = () => {
// // //     setIsDark(!isDark);
// // //   };

// // //   return (
// // //     <div className={isDark ? "dark-mode" : null}>
// // //       <StyleProvider value={{isDark: isDark, changeTheme: changeTheme}}>
// // //         {isShowingSplashAnimation && splashScreen.enabled ? (
// // //           <SplashScreen />
// // //         ) : (
// // //           <>
// // //             <Header />
// // //             <Greeting />
// // //             <Skills />
// // //             <StackProgress />
// // //             <Education />
// // //             <WorkExperience />
// // //             <Projects />
// // //             <StartupProject />
// // //             <Achievement />
// // //             <Blogs />
// // //             <Talks />
// // //             <Twitter />
// // //             <Podcast />
// // //             <Profile />
// // //             <Footer />
// // //             <ScrollToTopButton />
// // //           </>
// // //         )}
// // //       </StyleProvider>
// // //     </div>
// // //   );
// // // };

// // // export default Main;

// // // src/containers/Main.js
// // import React, {useEffect, useState} from "react";
// // import Header from "../components/header/Header";
// // import Greeting from "./greeting/Greeting";
// // import Skills from "./skills/Skills";
// // import StackProgress from "./skillProgress/skillProgress";
// // import WorkExperience from "./workExperience/WorkExperience";
// // import Projects from "./projects/Projects";
// // import StartupProject from "./StartupProjects/StartupProject";
// // import Achievement from "./achievement/Achievement";
// // import Blogs from "./blogs/Blogs";
// // import Footer from "../components/footer/Footer";
// // import Talks from "./talks/Talks";
// // import Podcast from "./podcast/Podcast";
// // import Education from "./education/Education";
// // import ScrollToTopButton from "./topbutton/Top";
// // import Twitter from "./twitter-embed/twitter";
// // import Profile from "./profile/Profile";
// // import SplashScreen from "./splashScreen/SplashScreen";
// // import {splashScreen} from "../portfolio";
// // import {StyleProvider} from "../contexts/StyleContext";
// // import {useLocalStorage} from "../hooks/useLocalStorage";
// // import "./Main.scss";
// // import Project1 from "../pages/projects/Project1";
// // import Project4 from "../pages/projects/Project4";

// // const Main = () => {
// //   const darkPref = window.matchMedia("(prefers-color-scheme: dark)");
// //   const [isDark, setIsDark] = useLocalStorage("isDark", darkPref.matches);
// //   const [isShowingSplashAnimation, setIsShowingSplashAnimation] =
// //     useState(true);
// //   const [currentPage, setCurrentPage] = useState("main");

// //   // 페이지 상태 변경 시 URL 업데이트, 그리고 popstate 이벤트 리스너 추가
// //   useEffect(() => {
// //     // 1. 뒤로가기/앞으로가기 버튼을 눌렀을 때 실행될 함수
// //     const handlePopState = () => {
// //       const path = window.location.pathname.replace("/", "");
// //       if (path === "project1" || path === "project4") {
// //         setCurrentPage(path);
// //       } else {
// //         setCurrentPage("main");
// //       }
// //     };

// //     // 2. popstate 이벤트 리스너를 추가
// //     window.addEventListener("popstate", handlePopState);

// //     // 3. 현재 URL에 따라 페이지 상태를 초기화
// //     const initialPath = window.location.pathname.replace("/", "");
// //     if (initialPath === "project1" || initialPath === "project4") {
// //       setCurrentPage(initialPath);
// //     } else {
// //       setCurrentPage("main");
// //     }

// //     // 4. 컴포넌트가 언마운트될 때 이벤트 리스너 제거
// //     return () => {
// //       window.removeEventListener("popstate", handlePopState);
// //     };
// //   }, []);

// //   // 이펙트 훅을 하나로 합칩니다.
// //   useEffect(() => {
// //     if (splashScreen.enabled) {
// //       const splashTimer = setTimeout(
// //         () => setIsShowingSplashAnimation(false),
// //         splashScreen.duration
// //       );
// //       return () => {
// //         clearTimeout(splashTimer);
// //       };
// //     }
// //   }, []);

// //   const changeTheme = () => {
// //     setIsDark(!isDark);
// //   };

// //   const changePage = pageName => {
// //     setCurrentPage(pageName);
// //     window.history.pushState(
// //       null,
// //       "",
// //       pageName === "main" ? "/" : `/${pageName}`
// //     );
// //   };

// //   return (
// //     <div className={isDark ? "dark-mode" : null}>
// //       <StyleProvider value={{isDark: isDark, changeTheme: changeTheme}}>
// //         {isShowingSplashAnimation && splashScreen.enabled ? (
// //           <SplashScreen />
// //         ) : (
// //           <>
// //             <Header />

// //             {currentPage === "main" && (
// //               <>
// //                 <Greeting />
// //                 <Skills />
// //                 <StackProgress />
// //                 <Education />
// //                 <WorkExperience />
// //                 <Projects />
// //                 <StartupProject changePage={changePage} />
// //                 <Achievement />
// //                 <Blogs />
// //                 <Talks />
// //                 <Twitter />
// //                 <Podcast />
// //                 <Profile />
// //               </>
// //             )}
// //             {currentPage === "project1" && <Project1 changePage={changePage} />}
// //             {currentPage === "project4" && <Project4 changePage={changePage} />}

// //             <Footer />
// //             <ScrollToTopButton />
// //           </>
// //         )}
// //       </StyleProvider>
// //     </div>
// //   );
// // };

// // export default Main;

// // src/containers/Main.js
// import React, { useEffect, useState } from "react";
// import Header from "../components/header/Header";
// import Greeting from "./greeting/Greeting";
// import Skills from "./skills/Skills";
// import StackProgress from "./skillProgress/skillProgress";
// import WorkExperience from "./workExperience/WorkExperience";
// import Projects from "./projects/Projects";
// import StartupProject from "./StartupProjects/StartupProject";
// import Achievement from "./achievement/Achievement";
// import Blogs from "./blogs/Blogs";
// import Footer from "../components/footer/Footer";
// import Talks from "./talks/Talks";
// import Podcast from "./podcast/Podcast";
// import Education from "./education/Education";
// import ScrollToTopButton from "./topbutton/Top";
// import Twitter from "./twitter-embed/twitter";
// import Profile from "./profile/Profile";
// import SplashScreen from "./splashScreen/SplashScreen";
// import { splashScreen } from "../portfolio";
// import { StyleProvider } from "../contexts/StyleContext";
// import { useLocalStorage } from "../hooks/useLocalStorage";
// import "./Main.scss";
// import Project1 from "../pages/projects/Project1";
// import Project4 from "../pages/projects/Project4";

// const Main = () => {
//   const darkPref = window.matchMedia("(prefers-color-scheme: dark)");
//   const [isDark, setIsDark] = useLocalStorage("isDark", darkPref.matches);
//   const [isShowingSplashAnimation, setIsShowingSplashAnimation] = useState(true);
//   const [currentPage, setCurrentPage] = useState("main");

//   // 페이지 상태 변경 시 URL 업데이트, 그리고 popstate 이벤트 리스너 추가
//   useEffect(() => {
//     const handlePopState = () => {
//       const path = window.location.pathname.replace("/", "");
//       if (path === "project1" || path === "project4") {
//         setCurrentPage(path);
//       } else {
//         setCurrentPage("main");
//       }
//     };

//     window.addEventListener("popstate", handlePopState);

//     const initialPath = window.location.pathname.replace("/", "");
//     if (initialPath === "project1" || initialPath === "project4") {
//       setCurrentPage(initialPath);
//     } else {
//       setCurrentPage("main");
//     }

//     return () => {
//       window.removeEventListener("popstate", handlePopState);
//     };
//   }, []);

//   useEffect(() => {
//     if (splashScreen.enabled) {
//       const splashTimer = setTimeout(
//         () => setIsShowingSplashAnimation(false),
//         splashScreen.duration
//       );
//       return () => {
//         clearTimeout(splashTimer);
//       };
//     }
//   }, []);

//   const changeTheme = () => {
//     setIsDark(!isDark);
//   };

//   const changePage = (pageName) => {
//     setCurrentPage(pageName);
//     window.history.pushState(
//       null,
//       "",
//       pageName === "main" ? "/" : `/${pageName}`
//     );
//   };

//   // ✅ currentPage가 바뀔 때마다 스크롤을 최상단으로 이동
//   useEffect(() => {
//     window.scrollTo(0, 0);
//   }, [currentPage]);

//   return (
//     <div className={isDark ? "dark-mode" : null}>
//       <StyleProvider value={{ isDark: isDark, changeTheme: changeTheme }}>
//         {isShowingSplashAnimation && splashScreen.enabled ? (
//           <SplashScreen />
//         ) : (
//           <>
//             <Header />

//             {currentPage === "main" && (
//               <>
//                 <Greeting />
//                 <Skills />
//                 <StackProgress />
//                 <Education />
//                 <WorkExperience />
//                 <Projects />
//                 <StartupProject changePage={changePage} />
//                 <Achievement />
//                 <Blogs />
//                 <Talks />
//                 <Twitter />
//                 <Podcast />
//                 <Profile />
//               </>
//             )}
//             {currentPage === "project1" && <Project1 changePage={changePage} />}
//             {currentPage === "project4" && <Project4 changePage={changePage} />}

//             <Footer />
//             <ScrollToTopButton />
//           </>
//         )}
//       </StyleProvider>
//     </div>
//   );
// };
import React, { useEffect, useState } from "react";
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
// 기존 프로젝트 컴포넌트 import
import Project1 from "../pages/projects/Project1";
import Project4 from "../pages/projects/Project4";
// GmmEmAssignmentPage 컴포넌트를 올바른 경로에서 import 합니다.
import GmmEmAssignmentPage from "../pages/projects/GmmEmAssignmentPage";

const Main = () => {
  const darkPref = window.matchMedia("(prefers-color-scheme: dark)");
  const [isDark, setIsDark] = useLocalStorage("isDark", darkPref.matches);
  const [isShowingSplashAnimation, setIsShowingSplashAnimation] = useState(true);
  const [currentPage, setCurrentPage] = useState("main");

  useEffect(() => {
    const handlePopState = () => {
      // 새로운 경로 'gmm-em-assignment'를 추가합니다.
      const path = window.location.pathname.replace("/", "");
      if (path === "project1" || path === "project4" || path === "gmm-em-assignment") {
        setCurrentPage(path);
      } else {
        setCurrentPage("main");
      }
    };

    window.addEventListener("popstate", handlePopState);

    const initialPath = window.location.pathname.replace("/", "");
    // 초기 경로 확인 로직에도 'gmm-em-assignment'를 추가합니다.
    if (initialPath === "project1" || initialPath === "project4" || initialPath === "gmm-em-assignment") {
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
      return () => {
        clearTimeout(splashTimer);
      };
    }
  }, []);

  const changeTheme = () => {
    setIsDark(!isDark);
  };

  const changePage = (pageName) => {
    setCurrentPage(pageName);
    window.history.pushState(
      null,
      "",
      pageName === "main" ? "/" : `/${pageName}`
    );
  };

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [currentPage]);

  return (
    <div className={isDark ? "dark-mode" : null}>
      <StyleProvider value={{ isDark: isDark, changeTheme: changeTheme }}>
        {isShowingSplashAnimation && splashScreen.enabled ? (
          <SplashScreen />
        ) : (
          <>
            <Header />

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
                {/* ✅ Blogs 컴포넌트에 changePage를 props로 넘겨줍니다. */}
                <Blogs changePage={changePage} />
                <Talks />
                <Twitter />
                <Podcast />
                <Profile />
              </>
            )}
            {currentPage === "project1" && <Project1 changePage={changePage} />}
            {currentPage === "project4" && <Project4 changePage={changePage} />}
            {/* ✅ 새로운 과제 페이지를 추가합니다. */}
            {currentPage === "gmm-em-assignment" && <GmmEmAssignmentPage changePage={changePage} />}

            <Footer />
            <ScrollToTopButton />
          </>
        )}
      </StyleProvider>
    </div>
  );
};

export default Main;