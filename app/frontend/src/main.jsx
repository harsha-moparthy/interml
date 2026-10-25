import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./tokens.css";
import "./styles.css";
import "./shell.css";
import "./browse.css";
import "./study.css";
import "./article.css";
import "./progress.css";

createRoot(document.getElementById("root")).render(<App />);
