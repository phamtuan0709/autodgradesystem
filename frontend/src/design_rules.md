# Project-Specific Design Rules (Light Theme Edition)

These design rules establish the visual standards for the HybridASAG Grader Web UI. Following these rules is mandatory to maintain absolute visual consistency across all components and views.

---

## 1. Typography & Hierarchy
- **Font Family**: Inter, Outfit, or standard clean system sans-serif.
- **Font Weights (Exactly 2)**:
  - `font-normal` (400) - For body text, descriptions, table cells, and regular labels.
  - `font-semibold` (600) - For headings, titles, active tabs, buttons, and important scores.
  - *No other font weights (e.g. 300, 500, 700, 800) are allowed.*
- **Text Colors (Exactly 2)**:
  - **Primary Text**: `text-slate-800` (e.g. `#1e293b` for high contrast, clean readability).
  - **Muted Text**: `text-slate-500` (e.g. `#64748b` for secondary descriptions and labels).
  - *Exceptions are only permitted for functional semantic status/error badges.*

## 2. Color Palette & Backgrounds
- **Primary Background**: `bg-slate-50` (soft, clean off-white canvas) with optional very light gradients (e.g., slate/indigo tint).
- **Surface / Card Background**: Pure white `bg-white` with rounded corners (`rounded-2xl`), a light border (`border-slate-200/80`), and a subtle shadow (`shadow-sm hover:shadow-md transition-shadow`).
- **Interactive States**: Hover state changes use clean background transition shifts (`hover:bg-slate-50`) and soft shadows.
- **Borders**: Thin, clean divisions using `border-slate-100` or `border-slate-200`.

## 3. Status & Error Badges (Distinct Colors)
To ensure error types are visually distinguishable at a glance, we use a distinct, soft pastel color palette for each status and issue label:
- **Correct**: Soft emerald background (`bg-emerald-50`), green-600 text, emerald-200 border.
- **Partially Correct**: Soft amber background (`bg-amber-50`), amber-600 text, amber-200 border.
- **Incorrect**: Soft rose background (`bg-rose-50`), rose-600 text, rose-200 border.
- **Missing Concepts**: Soft blue background (`bg-blue-50`), blue-600 text, blue-200 border.
- **Factual Error**: Soft red background (`bg-red-50`), red-600 text, red-200 border.
- **Logical Error**: Soft purple background (`bg-purple-50`), purple-600 text, purple-200 border.
- **Vague Expression**: Soft slate background (`bg-slate-100`), slate-600 text, slate-300 border.
- **Grammar Error**: Soft orange background (`bg-orange-50`), orange-600 text, orange-200 border.
- **Off-Topic**: Soft fuchsia background (`bg-fuchsia-50`), fuchsia-600 text, fuchsia-200 border.
- **Incomplete**: Soft yellow background (`bg-yellow-50`), yellow-600 text, yellow-200 border.

## 4. Icons & Visual Weights
- **Icon stroke-width** MUST adapt to adjacent text:
  - Adjacent to `font-normal` text: Use Lucide's default `strokeWidth={1.5}` or `strokeWidth={2}`.
  - Adjacent to `font-semibold` text: Use `strokeWidth={2.5}` to match the visual weight.
- **Spacing**: Maintain a consistent `gap-2` between icons and adjacent text labels.
