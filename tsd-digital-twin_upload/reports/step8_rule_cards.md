# TSD twin - alert rule cards (Step 8)

Every rule works on MEASURED tags and the gap between the measured and the physics-predicted
value. "Normal level" means the gap seen on healthy batches, learned from 247 batches
(models/runtime_calibration.json): HSD power +3.0 % vs physics, product
+1.5 degC vs measured-power physics, implied U 81 % of the
physics U. Every number below comes from SIMULATED data; refit the normal levels and re-tune the
thresholds on the first real batches (Step 9).

---

## TEMP_PREALARM - product temperature will reach the limit
**Watches:** the 5-min-ahead temperature forecast (physics + residual model + the batch's last 5 min)
and the measured temperature.
**Fires when:** forecast or measured temperature > limit - 3 degC, held 30 s.
**Also reports:** breach ETA = first crossing of the forecast curve shifted by the current residual
(fading with a 20-min time constant), and the likely cause (see evidence list).
**Achieved:** 22 of 22 breaches pre-alarmed, lead time median 15 min,
minimum 10 min (target >= 5); 33 watch flags on batches that stayed under the limit,
of which 23 peaked within 2 degC of it.
**Operator response:** reduce HSD speed 50-100 rpm or open the chilled-water valve; check the inlet temperature.
**Would fire wrongly on real data if:** the RTD is out of calibration (cross-check the now-cast gap), or the
recipe's limit in MES is wrong.

## LOW_CW_FLOW - chilled-water flow below set-point
**Watches:** measured flow / set-point.  **Fires when:** < 76 % for 60 s.
Achieved on the simulated year: 19 of 19 labelled batches caught (100 %), 0 false flags on 311 clearly healthy batches (0.0 per 100).
**Operator response:** check the strainer and the control valve; expect a hotter batch - the twin re-forecasts.
**Would fire wrongly if:** the flow meter drifts or the set-point tag is not updated when the operator changes it.

## OVERCHARGE - more material in the vessel than the recipe says
**Watches:** load-cell mass / recipe nominal (after charging).  **Fires when:** > 103.5 % for 60 s.
Achieved on the simulated year: 9 of 9 labelled batches caught (100 %), 0 false flags on 321 clearly healthy batches (0.0 per 100).
**Operator response:** confirm the dosing record; a 7-15 % overcharge lowers kWh/t but lengthens the grind and
can exceed the working volume.
**Would fire wrongly if:** the load cells are not tared between batches or the nominal mass in MES is stale.

## BLADE_WEAR - sawtooth blade losing its edge
**Watches:** HSD electrical power vs the physics power at the ACTUAL speed and viscosity, rolled over 5 min,
relative to the normal level.  **Fires when:** more than 2.5 % below normal for 60 s,
after 5 min of high-speed dispersion.  **Corroboration:** lab grind behind the physics prediction.
Achieved on the simulated year: 84 of 84 labelled batches caught (100 %), 0 false flags on 150 clearly healthy batches (0.0 per 100). Batches in the grey zone (wear 10-18 %, not yet labelled) flagged: 25 of
96 - early warnings, not errors.
**Cross-batch tracker:** CUSUM on the per-batch power residual with slack 0.5 % and decision interval 15.0 %,
plus a 30-batch trend giving "batches until the alarm level".
**Operator response:** plan a blade inspection; expect longer grinds and more extensions until it is changed.
**Would fire wrongly if:** the viscosity lot is unusually thin (check the anchor power residual), the VFD power
estimate drifts, or the blade is run at a different height.

## JACKET_FOULING - jacket transferring less heat than normal
**Watches:** (a) the implied overall U from the measured jacket duty, Q = m_dot cp (T_out - T_in), inverted through the
jacket equation, vs the physics U, relative to the normal level - only when the product is >= 5 degC above the
inlet; (b) the product temperature vs the measured-power physics, relative to the normal gap.
**Fires when:** rule variant 'u' - (a) > 4.5 % U drop, (b) > 0.35 degC -
held 60 s, after 10 min of dispersion, with flow normal (so it is not LOW_CW_FLOW).
Variant chosen because it meets the targets with the highest detection; the other variant stays as evidence in the cause text.
Achieved on the simulated year: 93 of 106 labelled batches caught (88 %), 4 false flags on 139 clearly healthy batches (2.9 per 100).
**Cross-batch tracker:** CUSUM on the per-batch U drop; trend gives "batches until the alarm level".
**Operator response:** schedule a jacket chemical clean; until then plan summer batches with the what-if tool.
**Would fire wrongly if:** the jacket outlet RTD reads low, the flow meter reads high, or the inlet cycles strongly
within the averaging window.
