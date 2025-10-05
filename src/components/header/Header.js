// src/components/header/Header.js
import React, {useContext} from "react";
import Headroom from "react-headroom";
import "./Header.scss";
import ToggleSwitch from "../ToggleSwitch/ToggleSwitch";
import StyleContext from "../../contexts/StyleContext";
import {
  greeting,
  workExperiences,
  skillsSection,
  openSource,
  blogSection,
  talkSection,
  achievementSection,
  resumeSection,
  educationInfo,
  bigProjects
} from "../../portfolio";

function Header({ changePage }) {
  const {isDark} = useContext(StyleContext);
  const viewExperience = workExperiences.display;
  const viewOpenSource = openSource.display;
  const viewSkills = skillsSection.display;
  const viewAchievement = achievementSection.display;
  const viewBlog = blogSection.display;
  const viewTalks = talkSection.display;
  const viewResume = resumeSection.display;
  const viewEducationInfo = educationInfo.display;
  const viewbigProjects = bigProjects.display;

  // 클릭 시: 메뉴 닫고 메인으로 전환 + 해당 섹션 스크롤
  const navigateTo = (sectionId) => (e) => {
    e.preventDefault();
    const checkbox = document.getElementById("menu-btn");
    if (checkbox) checkbox.checked = false;
    changePage("main", sectionId);
  };

  const goHome = (e) => {
    e.preventDefault();
    const checkbox = document.getElementById("menu-btn");
    if (checkbox) checkbox.checked = false;
    changePage("main");
  };

  return (
    <Headroom>
      <header className={isDark ? "dark-menu header" : "header"}>
        <a href="/" className="logo" onClick={goHome}>
          <span className="grey-color"> &lt;</span>
          <span className="logo-name">{greeting.username}</span>
          <span className="grey-color">&gt;</span>
        </a>

        <input className="menu-btn" type="checkbox" id="menu-btn" />
        <label className="menu-icon" htmlFor="menu-btn" style={{color: "white"}}>
          <span className={isDark ? "navicon navicon-dark" : "navicon"}></span>
        </label>

        <ul className={isDark ? "dark-menu menu" : "menu"}>
          {viewSkills && (
            <li><a href="/#skills" onClick={navigateTo("skills")}>Skills</a></li>
          )}
          {viewEducationInfo && (
            <li><a href="/#education" onClick={navigateTo("education")}>EducationInfo</a></li>
          )}
          {viewExperience && (
            <li><a href="/#experience" onClick={navigateTo("experience")}>Experiences</a></li>
          )}
          {viewbigProjects && (
            <li><a href="/#projects" onClick={navigateTo("projects")}>bigProjects</a></li>
          )}
          {viewOpenSource && (
            <li><a href="/#opensource" onClick={navigateTo("opensource")}>Open Source</a></li>
          )}
          {viewAchievement && (
            <li><a href="/#achievements" onClick={navigateTo("achievements")}>Achievements</a></li>
          )}
          {viewBlog && (
            <li><a href="/#blogs" onClick={navigateTo("blogs")}>Blogs</a></li>
          )}
          {viewTalks && (
            <li><a href="/#talks" onClick={navigateTo("talks")}>Talks</a></li>
          )}
          {viewResume && (
            <li><a href="/#resume" onClick={navigateTo("resume")}>Resume</a></li>
          )}
          <li><a href="/#contact" onClick={navigateTo("contact")}>Contact Me</a></li>

          <li>
            {/* eslint-disable-next-line jsx-a11y/anchor-is-valid */}
            <a><ToggleSwitch /></a>
          </li>
        </ul>
      </header>
    </Headroom>
  );
}
export default Header;
