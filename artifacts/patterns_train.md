# Patterns — train

## braces
- regex: `\{[^}]*\}`
- match_count: 865

### examples
- `004a7dbd-57ce-46f8-9691-409be61c676e`: `KIŠIB ma-nu-ba-lúm-a-šur DUMU ṣí-lá-{d}IM KIŠIB šu-{d}EN.LÍL DUMU ma-nu-ki-a-šur KIŠIB MA`
- `004a7dbd-57ce-46f8-9691-409be61c676e`: `IŠIB ma-nu-ba-lúm-a-šur DUMU ṣí-lá-{d}IM KIŠIB šu-{d}EN.LÍL DUMU ma-nu-ki-a-šur KIŠIB MAN-a-šur DUMU a-`
- `009fb838-8038-42bc-ad34-5f795b3840ee`: `KIŠIB šu-{d}EN.LÍL DUMU šu-ku-bi-im KIŠIB ṣí-lu-lu DUMU ú-ku i`
- `00aa1c55-c80c-4346-a159-73ad43ab0ff7`: `-kà-ni a ma-ma-an lá tù-šé-ri <gap> GÍN KÙ.BABBAR {d}UTU-tap-pá-i ub-lá-ki-im 1 GÍN KÙ.GI ù <gap> GÍN K`
- `00f0d841-eb7a-46f8-86fc-bf9fd7d52cbf`: `pá-ni-ma a-šar ta-ma-ḫa-ra-ni KÙ.BABBAR ša É a-lim{ki} šu-ta-aṣ-bi-ta-ma ma-lá a-bu-ku-nu ḫa-bu-lu KÙ.BA`
- `0126cd13-acf7-4cd5-8373-f1e7b54d824e`: `dí-a a-bu-ša-lim al-qé ša-lim-a-šur ša ni-iš a-lim{ki} úz-na-tí-ni il₅-pu-ut um-ma ša-lim-a-šur-ma a-šar`
- `017c5be4-f459-4ef5-a045-5933e79c01f9`: `u-um ša 10 ma-na KÙ.BABBAR ša PÚZUR-a-šur ù PÚZUR-{d}IŠKUR ṭup-pu-um ša 4 ma-na KÙ.BABBAR ša PÚZUR-a-šu`
- `020aa8bd-69a4-43a9-8120-efd9e587bb1c`: `GÚ URUDU ma-sí-am ša-bu-ra-am ḫu-ub-lam-ma PUZUR₄-{d}UTU li-pu-ul-kà 5 GÚ URUDU ma-sí-am ša-bu-ra-am en`
- `020aa8bd-69a4-43a9-8120-efd9e587bb1c`: `-am en-um-a-šur a-na i-ku-pí-a iḫ-bu-ul-ma PUZUR₄-{d}UTU en-um-a-šur e-pu-ul um-ma PUZUR₄-{d}UTU-ma a-n`
- `020aa8bd-69a4-43a9-8120-efd9e587bb1c`: `ma PUZUR₄-{d}UTU en-um-a-šur e-pu-ul um-ma PUZUR₄-{d}UTU-ma a-na 15 u₄-me-e 5 GÚ URUDU-a-kà a-na-ku a-š`

## brackets
- regex: `\[[^\]]*\]`
- match_count: 0

_examples: none_

## parens
- regex: `\([^)]*\)`
- match_count: 5

### examples
- `241862c4-7bd3-4ad9-a468-510bf7bcd747`: `(5 broken lines) <gap> i <gap> ší ri <gap> ša ší-bu-tí-šu <gap> ma`
- `5c5db0da-18fc-41f3-b744-41f23fb2db09`: ` 1 GÍN KÙ.BABBAR ša-áš-qí-lá-šu lá ta-ga-mì-lá-šu (5 broken lines) <gap> ma-na <gap>-i-dí DUMU lá+lá-<gap> KÙ.BABBAR`
- `659e367b-9c69-4945-90ff-c0fae71c41cc`: `GÍN AN.NA ša ku-ku-ú 5 GÍN AN.NA a šé-né-šu <gap> (broken line) 17 TÚG ku-ta-ni-šu 1 TÚG a-na ni-is-ḫa-tim il₅-qé`
- `872e8ca9-3b92-4973-aa4a-bb1311a4a183`: `u ù šu-nu-nu-ma a-na en-nam-a-šùr qí-bí-ma 50 TÚG (ḪI)ku-ta-ni 25 ma-na AN.NA 0.3333 ma-na 0.6666 GÍN KÙ`
- `944fb854-c4bf-41e9-9d7a-e235b706414d`: `(4 broken lines) <gap> an-ma-ma tí-ir-tí i-sú-ri-ik a-na ṣé-ri-a i`

## angle
- regex: `<[^>]*>`
- match_count: 3224

### examples
- `00aa1c55-c80c-4346-a159-73ad43ab0ff7`: `-na lá be-tim i-tù-ar a-pu-tum a-na en-um-a-šùr i-<gap>-ni-ma e ší-na ga <gap> ša lá ta-ḫa-dì-ri a-na IŠT`
- `00aa1c55-c80c-4346-a159-73ad43ab0ff7`: `-pu-tum a-na en-um-a-šùr i-<gap>-ni-ma e ší-na ga <gap> ša lá ta-ḫa-dì-ri a-na IŠTAR-lá-ma-sí qí-bi₄-ma š`
- `00aa1c55-c80c-4346-a159-73ad43ab0ff7`: ` lu ša-ṣú-ru pì-ri-kà-nu ša ma-tí ù tí-bu-lá ma-a-<gap> iš-ta-ú-mu-ni a-dí en-um-a-šùr i-lá-kà-ni a ma-ma`
- `00aa1c55-c80c-4346-a159-73ad43ab0ff7`: `-dí en-um-a-šùr i-lá-kà-ni a ma-ma-an lá tù-šé-ri <gap> GÍN KÙ.BABBAR {d}UTU-tap-pá-i ub-lá-ki-im 1 GÍN K`
- `00aa1c55-c80c-4346-a159-73ad43ab0ff7`: `.BABBAR {d}UTU-tap-pá-i ub-lá-ki-im 1 GÍN KÙ.GI ù <gap> GÍN KÙ.BABBAR i-ku-pì-a ub-lá-ki-im`
- `01cc1406-fc42-4252-aeae-6b4538b069c5`: `-im 30 ma-na URUDU IŠTAR-pí-lá-aḫ a-na ar-be-e-šu <gap> ù-kà-pu-ú ša-pì-ú-tim 9 ma-na URUDU SIG₅ i ší-im `
- `020aa8bd-69a4-43a9-8120-efd9e587bb1c`: `DINGIR i-ša-ḫi-iṭ-ma ḫa-muš-tum ša {d}IM-ṣú-lu-li <gap> ITU.KAM ku-zal-li li-mu-um ṣí-lu-lu um-ma i-ku-pí`
- `02351a98-b66c-42eb-90a7-11d834afe8e8`: `-šùr ù a-la-ḫu-um-ma ta-áš-pu-ra-am um-ma a-ta-ma <gap> ku-ta-nu a-ḫa-ma 92 TÚG a-ba-ar-ni-ú <gap>-a-šùr `
- `02351a98-b66c-42eb-90a7-11d834afe8e8`: `-ta-ma <gap> ku-ta-nu a-ḫa-ma 92 TÚG a-ba-ar-ni-ú <gap>-a-šùr qí-ip-tim <gap> ku-ta-ni <gap> bu-ra-e <gap`
- `02351a98-b66c-42eb-90a7-11d834afe8e8`: `a-ḫa-ma 92 TÚG a-ba-ar-ni-ú <gap>-a-šùr qí-ip-tim <gap> ku-ta-ni <gap> bu-ra-e <gap> ta-áš-pu-ra-ni <gap>`

## pipe
- regex: `\|[^|]*\|`
- match_count: 0

_examples: none_

## at_line
- regex: `(^|\n)\s*@\S+`
- match_count: 0

_examples: none_

## dollar_line
- regex: `(^|\n)\s*\$\s*\S+`
- match_count: 0

_examples: none_

## hash_line
- regex: `(^|\n)\s*#\S+`
- match_count: 0

_examples: none_

## percent_code
- regex: `%[a-zA-Z]{1,4}`
- match_count: 0

_examples: none_

## subscript_digits
- regex: `[₀₁₂₃₄₅₆₇₈₉]`
- match_count: 3668

### examples
- `004a7dbd-57ce-46f8-9691-409be61c676e`: `a-šur DUMU a-ta-a 0.3333 ma-na 2 GÍN KÙ.BABBAR SIG₅ i-ṣé-er PUZUR₄-a-šur DUMU a-ta-a a-lá-ḫu-um i-šu `
- `004a7dbd-57ce-46f8-9691-409be61c676e`: `-a 0.3333 ma-na 2 GÍN KÙ.BABBAR SIG₅ i-ṣé-er PUZUR₄-a-šur DUMU a-ta-a a-lá-ḫu-um i-šu iš-tù ḫa-muš-ti`
- `0064939c-59b9-4448-a63d-34612af0a1b5`: `1 TÚG ša qá-tim i-tur₄-DINGIR il₅-qé`
- `0064939c-59b9-4448-a63d-34612af0a1b5`: `1 TÚG ša qá-tim i-tur₄-DINGIR il₅-qé`
- `009fb838-8038-42bc-ad34-5f795b3840ee`: `ùr i-dí-in um-ma šu-ut-ma i-ṣí-ba-at KÙ-pì-a li-il₅-qé`
- `00aa1c55-c80c-4346-a159-73ad43ab0ff7`: `tum-ma a-na IŠTAR-lá-ma-sí ù ni-ta-aḫ-šu-šar qí-bi₄-ma mì-šu ša ta-áš-pu-ra-ni-ni um-ma a-tí-na-ma É-`
- `00aa1c55-c80c-4346-a159-73ad43ab0ff7`: ` <gap> ša lá ta-ḫa-dì-ri a-na IŠTAR-lá-ma-sí qí-bi₄-ma šu-ma a-ḫa-tí a-ta li-ba-am dì-ni-ší-im lá ta-`
- `00aa1c55-c80c-4346-a159-73ad43ab0ff7`: `ni-ší-im lá ta-ḫa-da-ar a-na ni-ta-aḫ-šu-šar qí-bi₄-ma TÚG-pì-ri-kà-ni ša e-zi-bu na-pí-ší-šu-nu ù ṭu`
- `00f0d841-eb7a-46f8-86fc-bf9fd7d52cbf`: ` ù lá-ma-sí-ma a-na en-um-a-šùr ù a-lá-ḫi-im qí-bi₄-ma a-ma-la na-áš-pé-er-tí-ku-nu ra-bi-ṣa-am ni-ḫu`
- `00f0d841-eb7a-46f8-86fc-bf9fd7d52cbf`: `-ṣa-am ni-ḫu-za-ku-nu-tí a-bi-a DUMU be-e-be ra-bi₄-iṣ-ni i-na ša-am-ší ša ra-bi-ṣú-um e-ra-ba-ni iḫ-`
