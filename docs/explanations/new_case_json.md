# `examples/new_case.json`: explained

**Role:** an example input for `student-risk predict --input examples/new_case.json`. It is used
in the README and for the live demo.

**What it is:**

- **One JSON object** describing **one invented student**, with exactly the 30 feature fields the
  model was trained on. It is not a row copied from the dataset.
- There is no G1, G2 or G3: those are not known early, and `predict` would reject them as
  "unknown fields".
- Every value is within the ranges and categories of the real data (checked on the download):
  - text fields use the dataset's codes, for example `"school": "MS"` (`GP` or `MS`) and
    `"Mjob": "at_home"`;
  - numeric fields are JSON numbers, for example `"studytime": 1`, which is in the 1–4 range.

**Why this student:** the profile was chosen to look at risk. It has:

- 1 past failure;
- the lowest study time (1);
- no wish for higher education (`"higher": "no"`);
- 10 absences;
- frequent outings (`"goout": 4`).

So the demo shows a flagged case.

**Result with the trained model (checked):** probability of failing 0.496, threshold 0.140, so
the decision is **"AT RISK - flag for tutoring"**.

**Meaning of the codes:** they follow the UCI attribute list. For example `Medu` is the mother's
education, from 0 (none) to 4 (higher education). `famrel`, `freetime`, `goout`, `Dalc`, `Walc`
and `health` go from 1 (very low or bad) to 5 (very high or good). The codes in this paragraph
are **[from prior knowledge, to verify on the UCI page]**.

**To try a different student:** copy the file and change the values. Removing a field, adding an
unknown one, or writing a number as text gives a clear error message (see `predict.md`).
