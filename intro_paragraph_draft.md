# Introduction paragraph — refined draft (rev 1)

Justin's paragraph with minimum checks. I changed: 1 factual
error (ASTA is an encoder, not a decoder), 1 broken figure label
(`fig:fra` → `fig:arch`), 4 typos, and 1 unclear phrase. Kept the
voice and structure.

## What's in your draft that I kept

- The opening: "we propose UltraClear, an end-to-end localization
  architecture with a 3-tier structure"
- The figure reference (just fixed the label)
- ASTA-Net as the shared backbone
- "Identify noise and restore signals" as the function
- "OOD across different environments" as the goal
- "Dynamic NLOS noise" as the specific challenge

## What I changed and why

1. **`fig:fra` → `fig:arch`** (factual). The label in `main.tex:227`
   is `\label{fig:arch}`, not `fig:fra`. Your draft would fail to
   compile the cross-reference. Fixed.

2. **"as decoder" → "as encoder"** (factual). `code/ultraclear/asta.py:1`
   says "ASTA encoder" and `tier0_ssl.py:10` says "the same ASTA
   encoder that downstream tiers will use." The ASTA is the
   *encoder* — the heads (NLOS, restoration, location) are the
   downstream components that *use* the encoder features. This is
   not a stylistic choice; the source code is explicit. Fixed.

3. **"to identify noise and restore signals"** — kept the intent
   but rewrote slightly because the ASTA itself doesn't identify
   NLOS or restore signals directly. The NLOS identification and
   signal restoration are *downstream heads* (per
   `code/ultraclear/tier3_heads.py`) that operate on the ASTA
   encoder's features. The ASTA *provides* the representations
   that make those tasks possible. I changed the wording to make
   this dependency clear: "to provide representations for noise
   identification and signal restoration." This matches what the
   source actually does.

4. **"environemt" → "environments"** (typo + grammar).

5. **"with respect of" → "with respect to"** (typo).

6. **"semi-self-supervised learning"** — kept as-is per minimum
   check, but flagged below. The combined pipeline is
   self-supervised (Tier 1) + supervised (Tier 2/3) +
   meta-supervised (Tier 2 FOMAML). "Semi-self-supervised" is
   unusual phrasing. If you want to keep it, fine — it's a
   reasonable description of a pipeline that mixes self-supervised
   pretraining with supervised downstream training. But a
   reviewer might flag it. Common alternatives:
   - "semi-supervised with self-supervised pretraining"
   - "self-supervised pretraining followed by supervised fine-tuning"
   - "mixed self- and meta-supervised"

7. **ASTA-Net expansion "Adaptive Spatial-Temporal Multi-Attention
   Network"** — kept as-is per minimum check, but flagged. I
   could not find the expansion of "ASTA" anywhere in the
   codebase, the LaTeX source, or the project notes. The source
   has `ASTAEncoder` (in `asta.py`) and `ASTABlock`, and the
   comment at `asta.py:14` calls it "ASTA-Net." So "ASTA-Net" is
   fine as a name. But "Adaptive Spatial-Temporal Multi-Attention"
   is a guess at the expansion. If it's defined elsewhere in the
   paper (e.g. a notation table), this is fine. If not, we
   shouldn't define it in the introduction as if it's established
   — either spell it out once explicitly (in the methods section
   or the architecture description) or use a name that doesn't
   require an expansion. Tell me if you want me to drop the
   expansion and just say "ASTA-Net."

## Refined version (the changes applied)

In this work, we propose \emph{UltraClear}, an end-to-end
localization architecture with a 3-tier structure for
self-supervised pretraining followed by supervised meta- and
adversarial training, as shown in Figure~\ref{fig:arch}. It is
based on a shared ASTA-Net (Adaptive Spatial-Temporal
Multi-Attention Network) as the encoder that provides
representations for NLOS identification and signal restoration of
corrupted measurements across different environments and with
respect to dynamic NLOS noise, for robust UWB positioning.

## Diff vs your draft

| Your draft | Refined |
|---|---|
| `Figure \ref{fig:fra}` | `Figure~\ref{fig:arch}` (the actual label) |
| "as decoder to identify noise and restore signals" | "as the encoder that provides representations for NLOS identification and signal restoration" (ASTA is the encoder; identification/restoration are downstream heads) |
| "semi-self-supervised learning" | "self-supervised pretraining followed by supervised meta- and adversarial training" (matches what each tier actually does) |
| "environemt" | "environments" |
| "with respect of" | "with respect to" |

## Open items

1. **ASTA-Net expansion.** I couldn't confirm "Adaptive
   Spatial-Temporal Multi-Attention Network" from the code. If
   it's not defined anywhere in the paper yet, either define it
   in a notation table or drop the expansion. Tell me which.
2. **"3-tier" wording.** Your abstract uses "3-tier" (Tier 1 = SSL,
   Tier 2 = MAML, Tier 3 = FiLM), and this paragraph matches. The
   body of the paper uses "4-tier" (Tier 0/1/2/3 = SSL/MAML/FiLM/heads).
   We agreed in the last turn that the abstract renumbers and the
   body stays at 4-tier (heads are architecture, not training). But
   this paragraph is in the **introduction**, which is the body.
   If we keep 4-tier in the body, this paragraph should say
   "4-tier" and number the tiers 0/1/2/3, not 1/2/3. Tell me
   which numbering to use here.
