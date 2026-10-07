# Live integration probe

This file exists only to validate managed CI for trusted integration branches (#1286, stage C step 4 of the #1210 stage B work): it is implemented into the integration branch `refactor/live-probe` rather than `main`, and the `refactor/live-probe` branch is deleted afterwards, so nothing in this probe reaches `main`.
