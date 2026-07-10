# MuSiQue robustness qualitative examples

Selección automática de casos útiles para interpretar clean/noisy/adversarial.

## adversarial_all_rag_wrong

### musique_mc__0053

**Question:** The territory containing Bansaan Island is located at which island?

**Options:**

```text
A. Cebu Island
B. Panglao Island
C. Bohol Island
D. Siargao Island
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | C ✗ | C ✗ | C ✗ |
| noisy | C ✗ | C ✗ | C ✗ |
| adversarial | C ✗ | C ✗ | C ✗ |

**Detected patterns:** clean_all_rag_wrong, noisy_all_rag_wrong, adversarial_all_rag_wrong

---

### musique_mc__0061

**Question:** What event caused the Liberal Party of Australia's longest-serving leader to become Prime Minister?

**Options:**

```text
A. Lyons' death in 1939
B. Winning the 1949 federal election
C. Resignation of the prime minister in 1941
D. A party leadership challenge in 1939
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | B ✗ | B ✗ | B ✗ |
| noisy | B ✗ | A ✓ | B ✗ |
| adversarial | B ✗ | B ✗ | B ✗ |

**Detected patterns:** s2_recovers_noisy, clean_all_rag_wrong, noisy_only_s2_correct, adversarial_all_rag_wrong, noisy_s2_beats_s1

---

### musique_mc__0070

**Question:** The Move Ya Body song's band is named after who?

**Options:**

```text
A. Jennifer Lopez
B. Natalie Albino
C. Nicole Scherzinger
D. Nicole Albino
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | D ✗ | D ✗ | B ✓ |
| noisy | D ✗ | D ✗ | D ✗ |
| adversarial | D ✗ | D ✗ | D ✗ |

**Detected patterns:** s3_mc_regresses_noisy, s3_mc_regresses_adversarial, clean_only_s3_mc_correct, noisy_all_rag_wrong, adversarial_all_rag_wrong

---

## adversarial_only_s3_mc_correct

### musique_mc__0069

**Question:** What is the performer of Heartbeat named after?

**Options:**

```text
A. Natalie Merchant
B. Nina Simone
C. Natalie Albino
D. Nicole Albino
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | B ✗ | B ✗ | C ✓ |
| noisy | B ✗ | B ✗ | C ✓ |
| adversarial | D ✗ | B ✗ | C ✓ |

**Detected patterns:** clean_only_s3_mc_correct, noisy_only_s3_mc_correct, adversarial_only_s3_mc_correct

---

## adversarial_s1_beats_s2

### musique_mc__0021

**Question:** Where was the author of Hannibal and Scipio educated at?

**Options:**

```text
A. Pembroke College
B. Exeter College
C. Magdalen College
D. Oriel College
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | D ✗ | B ✓ | B ✓ |
| noisy | D ✗ | A ✗ | C ✗ |
| adversarial | B ✓ | D ✗ | C ✗ |

**Detected patterns:** s1_recovers_adversarial, s2_regresses_noisy, s2_regresses_adversarial, s3_mc_regresses_noisy, s3_mc_regresses_adversarial, noisy_all_rag_wrong, adversarial_only_s1_correct, clean_s2_beats_s1

---

### musique_mc__0024

**Question:** What county was Tim Dubois born in?

**Options:**

```text
A. Benton County
B. McDonald County
C. Lawrence County
D. Newton County
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | B ✓ | B ✓ | B ✓ |
| noisy | B ✓ | B ✓ | B ✓ |
| adversarial | B ✓ | C ✗ | B ✓ |

**Detected patterns:** s2_regresses_adversarial, clean_all_rag_correct, noisy_all_rag_correct, adversarial_s1_beats_s2

---

### musique_mc__0027

**Question:** What instrument is played by the person from The Blackout All-Stars?

**Options:**

```text
A. maraca
B. timbale
C. conga
D. bongo
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | C ✓ | C ✓ | B ✗ |
| noisy | B ✗ | C ✓ | B ✗ |
| adversarial | C ✓ | B ✗ | C ✓ |

**Detected patterns:** s1_regresses_noisy, s2_regresses_adversarial, s3_mc_recovers_adversarial, noisy_only_s2_correct, noisy_s2_beats_s1, adversarial_s1_beats_s2

---

## noisy_all_rag_wrong

### musique_mc__0010

**Question:** What league does the team that plays in Stadio Ciro Vigorito play for?

**Options:**

```text
A. Serie B
B. Lega Pro Prima Divisione
C. Serie D
D. Serie C
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | A ✗ | A ✗ | A ✗ |
| noisy | A ✗ | A ✗ | A ✗ |
| adversarial | A ✗ | A ✗ | A ✗ |

**Detected patterns:** clean_all_rag_wrong, noisy_all_rag_wrong, adversarial_all_rag_wrong

---

### musique_mc__0039

**Question:** Who is the child of the person who followed Tihomir of Serbia?

**Options:**

```text
A. Vukan Nemanjic
B. Miroslav of Hum
C. Stefan Nemanjic
D. Saint Sava
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | D ✓ | C ✗ | C ✗ |
| noisy | A ✗ | C ✗ | C ✗ |
| adversarial | A ✗ | C ✗ | C ✗ |

**Detected patterns:** s1_regresses_noisy, s1_regresses_adversarial, clean_only_s1_correct, noisy_all_rag_wrong, adversarial_all_rag_wrong, clean_s1_beats_s2

---

### musique_mc__0046

**Question:** Who was the sibling of Nannina de' Medici?

**Options:**

```text
A. Piero de' Medici
B. Giuliano de' Medici
C. Giovanni de' Medici
D. Lorenzo de' Medici
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | D ✗ | D ✗ | D ✗ |
| noisy | D ✗ | D ✗ | D ✗ |
| adversarial | D ✗ | D ✗ | D ✗ |

**Detected patterns:** clean_all_rag_wrong, noisy_all_rag_wrong, adversarial_all_rag_wrong

---

## noisy_s1_beats_s2

### musique_mc__0000

**Question:** Who is the spouse of the Green performer?

**Options:**

```text
A. Miquette Giraudy
B. Annie Haslam
C. Maggie Reilly
D. Gillian Gilbert
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | A ✓ | A ✓ | A ✓ |
| noisy | A ✓ | C ✗ | A ✓ |
| adversarial | A ✓ | D ✗ | A ✓ |

**Detected patterns:** s2_regresses_noisy, s2_regresses_adversarial, clean_all_rag_correct, noisy_s1_beats_s2, adversarial_s1_beats_s2

---

### musique_mc__0002

**Question:** What administrative territorial entity is the owner of Ciudad Deportiva located?

**Options:**

```text
A. Veracruz
B. Nuevo Leon
C. Tamaulipas
D. Chihuahua
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | C ✓ | C ✓ | C ✓ |
| noisy | C ✓ | B ✗ | C ✓ |
| adversarial | C ✓ | B ✗ | C ✓ |

**Detected patterns:** s2_regresses_noisy, s2_regresses_adversarial, clean_all_rag_correct, noisy_s1_beats_s2, adversarial_s1_beats_s2

---

### musique_mc__0008

**Question:** What company succeeded the owner of Empire Sports Network?

**Options:**

```text
A. Comcast Corporation
B. Charter Communications
C. Cox Communications
D. Time Warner Cable
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | A ✗ | D ✓ | D ✓ |
| noisy | D ✓ | A ✗ | D ✓ |
| adversarial | D ✓ | A ✗ | A ✗ |

**Detected patterns:** s1_recovers_noisy, s1_recovers_adversarial, s2_regresses_noisy, s2_regresses_adversarial, s3_mc_regresses_adversarial, adversarial_only_s1_correct, clean_s2_beats_s1, noisy_s1_beats_s2

---

## s2_regresses_adversarial

### musique_mc__0052

**Question:** Who is part of the band that performed Full Cooperation?

**Options:**

```text
A. Keith Murray
B. Busta Rhymes
C. Erick Sermon
D. Method Man
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | C ✓ | C ✓ | C ✓ |
| noisy | C ✓ | B ✗ | C ✓ |
| adversarial | C ✓ | B ✗ | C ✓ |

**Detected patterns:** s2_regresses_noisy, s2_regresses_adversarial, clean_all_rag_correct, noisy_s1_beats_s2, adversarial_s1_beats_s2

---

### musique_mc__0064

**Question:** Who founded the publisher of The Final Testament of the Holy Bible?

**Options:**

```text
A. John Murray
B. Larry Gagosian
C. Alfred A. Knopf
D. George Palmer Putnam
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | B ✓ | B ✓ | B ✓ |
| noisy | B ✓ | B ✓ | B ✓ |
| adversarial | B ✓ | A ✗ | B ✓ |

**Detected patterns:** s2_regresses_adversarial, clean_all_rag_correct, noisy_all_rag_correct, adversarial_s1_beats_s2

---

### musique_mc__0067

**Question:** In what era was the company known as the manufacturer of Agni-V founded?

**Options:**

```text
A. From the 1950s to the 1970s
B. From the 1890s to the 1910s
C. From the 1970s to the 1990s
D. From the 1920s to the 1940s
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | A ✓ | A ✓ | A ✓ |
| noisy | A ✓ | A ✓ | A ✓ |
| adversarial | A ✓ | C ✗ | A ✓ |

**Detected patterns:** s2_regresses_adversarial, clean_all_rag_correct, noisy_all_rag_correct, adversarial_s1_beats_s2

---

## s2_regresses_noisy

### musique_mc__0013

**Question:** What other county does the county where Imperial is located share a border with?

**Options:**

```text
A. Reeves County
B. Jeff Davis County
C. Crockett County
D. Upton County
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | A ✗ | C ✓ | A ✗ |
| noisy | A ✗ | A ✗ | A ✗ |
| adversarial | A ✗ | A ✗ | A ✗ |

**Detected patterns:** s2_regresses_noisy, s2_regresses_adversarial, clean_only_s2_correct, noisy_all_rag_wrong, adversarial_all_rag_wrong, clean_s2_beats_s1

---

### musique_mc__0016

**Question:** Where was Tyler MacDuff's child educated?

**Options:**

```text
A. Monroe High School
B. Blair High School
C. Wilson High School
D. Lincoln High School
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | B ✓ | B ✓ | B ✓ |
| noisy | B ✓ | C ✗ | B ✓ |
| adversarial | B ✓ | B ✓ | B ✓ |

**Detected patterns:** s2_regresses_noisy, clean_all_rag_correct, adversarial_all_rag_correct, noisy_s1_beats_s2

---

### musique_mc__0030

**Question:** What group was the performer of Be the One a member of?

**Options:**

```text
A. Four Tops
B. The Temptations
C. The Supremes
D. Jackson 5
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | D ✓ | D ✓ | D ✓ |
| noisy | D ✓ | C ✗ | D ✓ |
| adversarial | D ✓ | D ✓ | D ✓ |

**Detected patterns:** s2_regresses_noisy, clean_all_rag_correct, adversarial_all_rag_correct, noisy_s1_beats_s2

---

## s3_mc_regresses_adversarial

### musique_mc__0018

**Question:** Who is the child of Sigrid Eskilsdotter's child?

**Options:**

```text
A. Ture Pedersson Bielke
B. Svante Turesson Bielke
C. Svante Stensson Sture
D. Christina Gyllenstierna
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | B ✗ | B ✗ | C ✓ |
| noisy | B ✗ | B ✗ | C ✓ |
| adversarial | B ✗ | C ✓ | B ✗ |

**Detected patterns:** s2_recovers_adversarial, s3_mc_regresses_adversarial, clean_only_s3_mc_correct, noisy_only_s3_mc_correct, adversarial_only_s2_correct, adversarial_s2_beats_s1

---

### musique_mc__0020

**Question:** What record label is the performer of Almost Made Ya signed to?

**Options:**

```text
A. Interscope Records
B. Jive Records
C. Derrty Entertainment
D. Def Jam Recordings
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | C ✓ | C ✓ | C ✓ |
| noisy | B ✗ | C ✓ | C ✓ |
| adversarial | B ✗ | C ✓ | A ✗ |

**Detected patterns:** s1_regresses_noisy, s1_regresses_adversarial, s3_mc_regresses_adversarial, clean_all_rag_correct, adversarial_only_s2_correct, noisy_s2_beats_s1, adversarial_s2_beats_s1

---

### musique_mc__0056

**Question:** In which district was Ernie Watts born?

**Options:**

```text
A. North Somerset
B. East Hampshire
C. South Oxfordshire
D. West Berkshire
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | A ✗ | C ✗ | D ✓ |
| noisy | A ✗ | A ✗ | A ✗ |
| adversarial | A ✗ | C ✗ | A ✗ |

**Detected patterns:** s3_mc_regresses_noisy, s3_mc_regresses_adversarial, clean_only_s3_mc_correct, noisy_all_rag_wrong, adversarial_all_rag_wrong

---

## s3_mc_regresses_noisy

### musique_mc__0029

**Question:** Who is the father of Edward Baring, 1st Baron Revelstoke's father?

**Options:**

```text
A. Henry Baring, MP
B. Alexander Baring, 1st Baron Ashburton
C. Thomas George Baring, 1st Earl of Northbrook
D. Sir Francis Baring, 1st Baronet
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | D ✓ | A ✗ | D ✓ |
| noisy | D ✓ | D ✓ | A ✗ |
| adversarial | D ✓ | D ✓ | D ✓ |

**Detected patterns:** s2_recovers_noisy, s2_recovers_adversarial, s3_mc_regresses_noisy, adversarial_all_rag_correct, clean_s1_beats_s2

---

### musique_mc__0045

**Question:** In which county is Mark Dismore's birthplace located?

**Options:**

```text
A. Hancock County
B. Decatur County
C. Shelby County
D. Marion County
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | D ✗ | C ✗ | A ✓ |
| noisy | D ✗ | A ✓ | C ✗ |
| adversarial | D ✗ | A ✓ | A ✓ |

**Detected patterns:** s2_recovers_noisy, s2_recovers_adversarial, s3_mc_regresses_noisy, clean_only_s3_mc_correct, noisy_only_s2_correct, noisy_s2_beats_s1, adversarial_s2_beats_s1

---

### musique_mc__0048

**Question:** What league does the team that occupies the Rabat Ajax Football Ground belong to?

**Options:**

```text
A. Maltese Premier League
B. Maltese Challenge League
C. Maltese First Division
D. Maltese Second Division
```

| Condition | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | A ✓ | C ✗ | A ✓ |
| noisy | A ✓ | B ✗ | B ✗ |
| adversarial | B ✗ | B ✗ | A ✓ |

**Detected patterns:** s1_regresses_adversarial, s3_mc_regresses_noisy, noisy_only_s1_correct, adversarial_only_s3_mc_correct, clean_s1_beats_s2, noisy_s1_beats_s2

---

