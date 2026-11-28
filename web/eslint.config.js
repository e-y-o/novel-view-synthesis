import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "node_modules", "fixtures", "spike"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      // keep the rule set minimal: recommended only, plus unused vars with the usual `_` escape
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
);
