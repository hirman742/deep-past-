# A2 Promote Compare

- status: `manual_promote_recommended`
- selected candidate: `fallback_180`
- candidate type: post-hoc incumbent fallback on repeated generic chunk outputs
- note: this is a compare candidate, not a new trained checkpoint

## Scoreboard

- incumbent full-val / hard / anchor64: `14.3323 / 13.7161 / 16.5057`
- raw W-lite full-val / hard / anchor64: `19.9908 / 20.8360 / 23.5415`
- fallback_180 full-val / hard / anchor64: `19.9035 / 20.7888 / 23.5415`
- fallback_180 vs incumbent full-val delta: `+5.5713`
- fallback_180 vs raw W-lite full-val delta: `-0.0872`

## Health

- raw full-val health no_regression vs incumbent: `False`
- fallback_180 full-val health no_regression vs incumbent: `True`
- raw unique delta vs incumbent: `-1.5511`
- fallback_180 unique delta vs incumbent: `+0.0000`
- raw short delta vs incumbent: `+0.0816`
- fallback_180 short delta vs incumbent: `-0.4898`

## Changed Rows

- changed chunk rows: `58` / `1225` (`4.7347%`)
- changed parent rows: `29`
- changed original rows: `8`
- changed ratio rows: `23`
- changed short-aligned rows: `27`
- changed hard rows / parents: `25` / `12`

## Local Cost On Changed Rows

- raw changed-subset geom: `13.2549`
- fallback_180 changed-subset geom: `11.5200`
- incumbent changed-subset geom: `11.5200`
- interpretation: health is recovered by replacing a small set of high-frequency generic chunk outputs, but those changed rows are locally weaker than raw W-lite.

## Repeat Groups

- raw_count=`7` changed_rows=`7` changed_parents=`4` text=`To Ennam-Aššur, sealed by Ali-ahum and <gap>-Ištar.`
- raw_count=`6` changed_rows=`6` changed_parents=`3` text=`Investors of Šalim-Aššur may make demands Do this and dispatch the 2 tablets with verdicts of the City plus a copy of the testament, and hire an attorney and send Aluwa and the attorney as soon as possible`
- raw_count=`6` changed_rows=`6` changed_parents=`3` text=`It is important that you set out and leave the very day you hear my letter, before word <gap> It is important that you set out and leave the very day you hear my letter, before word <gap>`
- raw_count=`5` changed_rows=`5` changed_parents=`3` text=`The 14 minas <gap> 8.6666 shekels of silver is the price of tin, textiles and donkeys.`
- raw_count=`5` changed_rows=`5` changed_parents=`3` text=`To copper and 3 shekels of silver for his small goods, 3 shekels of silver <gap> I gave to Aluwa. I paid 4.5 shekels of silver, the fees from the City when he returned. <gap> minas of sickles, 3 shekels for his expenses…`
- raw_count=`5` changed_rows=`5` changed_parents=`3` text=`To Šalim-Aššur; sealed by Man-mahir.`
- raw_count=`4` changed_rows=`4` changed_parents=`2` text=`I gave 4.5 shekels of silver for the payment of his fees to Aluwa.`
- raw_count=`4` changed_rows=`4` changed_parents=`1` text=`In accordance with the tablet with a verdict of the City that Kakuwa brought Enna-Suen and Anuli must not raise claims to those goods against Šalim-Aššur.`
- raw_count=`4` changed_rows=`4` changed_parents=`2` text=`Send me 2 textiles, for I tore (mine), and 1 shekel of silver <gap> the container and the tabletbox <gap> they they entrusted to you <gap> as long as you do not see my eyes "`
- raw_count=`4` changed_rows=`4` changed_parents=`2` text=`Send me word with the first transport Do not focus on taking all of the silver Kura will not pay <gap>`

## Decision

- use `fallback_180` as the `A2_F_review` promote candidate
- keep raw W-lite as score ceiling reference, but do not promote it directly while health gate remains red
- do not treat this as a new trained model; it is a selective fallback post-process candidate
