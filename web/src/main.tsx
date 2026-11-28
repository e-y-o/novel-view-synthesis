import { render } from "preact";
import { App } from "./app";
import "./styles/base.css";
import "./styles/app.css";

const root = document.getElementById("app");
if (!root) throw new Error("#app element missing from index.html");
render(<App />, root);
