// import React from "react";
// import "./App.scss";
// import Main from "./containers/Main";

// function App() {
//   return (
//     <div>
//       <Main />
//     </div>
//   );
// }
// App.js
import {BrowserRouter as Router} from "react-router-dom";
import ScrollToTop from "./components/ScrollToTop";
import Main from "./containers/Main";

function App() {
  return (
    <Router>
      <ScrollToTop />
      <Main />
    </Router>
  );
}

export default App;
