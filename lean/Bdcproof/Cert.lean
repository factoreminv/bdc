/-
  Machine-checked core of the upper bound on the capacity of the binary deletion channel.

  PROVED HERE — the steps that are novel to this work, or where a direction error would
  silently invalidate the numeric result:

    * `cert_bound`, `cert_rate` (Step 9).  The certificate: a potential `h` satisfying a
      finite inequality bounds the reward of EVERY trajectory of EVERY length.  This is
      what turns a finite computer search into a theorem, and why the bound does not
      depend on the solver having converged.
    * `geom_sum_le_inv`, `geom_tail` (Step 7).  The geometric remainder that makes the
      truncated tail a genuine UPPER bound.  The original code summed finitely many terms,
      which bounds the tail from BELOW; the audit caught this.
    * `assembly` (Steps 1–9 combined).  Each truncation enters as its own hypothesis, so
      charging one in the wrong direction cannot typecheck.
    * `extend_in_d` (Step 10).  The arithmetic of pushing a bound at one `d` to all larger
      `d` through the cited monotonicity lemma.

  ASSUMED (as explicit hypotheses, not axioms of the file): the information-theoretic
  content of Steps 0–6 — the finite-length reduction of Fertonani–Duman, the
  Rahmati–Duman monotonicity, the entropy chain rule, "dropping conditioning increases
  entropy", and the exact embedding identity.  These are standard or cited; formalising
  them needs a full development of Shannon entropy for random variables and of the
  channel itself.
-/
import Mathlib

namespace BDC

open Finset

/-! ### One-sided lower-bound certificate -/

/-- Retaining nonnegative segmentation contributions and replacing each by a certified
lower value preserves achievability.  This is the algebraic kernel consumed by a rational
quadrature certificate. -/
theorem retained_corrections_lower {i : Type*} [Fintype i]
    (mass kernel lower : i → ℝ)
    (hmass : ∀ j, 0 ≤ mass j) (hlower : ∀ j, lower j ≤ kernel j) :
    (∑ j, mass j * lower j) ≤ ∑ j, mass j * kernel j := by
  exact Finset.sum_le_sum fun j _ =>
    mul_le_mul_of_nonneg_left (hlower j) (hmass j)

/-- Assembly of a baseline PRC rate and a certified nonnegative correction. -/
theorem prc_lower_assembly (C : ℝ → ℝ) (d R R₀ corr : ℝ)
    (_hd : 0 ≤ d) (hd1 : d ≤ 1)
    (hbase : R₀ + corr ≤ R)
    (hprc : (1 - d) * R ≤ C d) :
    (1 - d) * (R₀ + corr) ≤ C d := by
  have hnonneg : 0 ≤ 1 - d := by linarith
  exact le_trans (mul_le_mul_of_nonneg_left hbase hnonneg) hprc

/-- Decimal arithmetic used by the Arb certificate.  Each input below is rounded in the
weak direction from the machine-produced ball enclosure. -/
theorem lower_decimal_arithmetic :
    (0.12415 : ℝ) ≤ 0.119876 + 0.888686 *
      (0.030455 + 0.037617 * 0.058937) / 6.780381 := by
  norm_num

/-- Capacity form of the certified decimal once the finite Arb inequalities are supplied. -/
theorem headline_lower_012415 (C : ℝ → ℝ) (d : ℝ) (_hd : 0 ≤ d) (hd1 : d ≤ 1)
    (hfinite : (1 - d) *
      (0.119876 + 0.888686 * (0.030455 + 0.037617 * 0.058937) / 6.780381) ≤ C d) :
    0.12415 * (1 - d) ≤ C d := by
  have hn : 0 ≤ (1 : ℝ) - d := by linarith
  have hmul := mul_le_mul_of_nonneg_right lower_decimal_arithmetic hn
  exact le_trans hmul (by simpa [mul_comm] using hfinite)

/-! ### Step 9 — the certificate -/

/-- **Telescoping certificate.**  If `h` makes every one-step reward at most `θ` after
correction, the accumulated reward along any trajectory is at most `n * θ` plus a boundary
term independent of `n`.  No optimality of `h` is needed. -/
theorem cert_bound {S : Type*} (next : S → Bool → S) (R : S → Bool → ℝ)
    (h : S → ℝ) (θ : ℝ) (hcert : ∀ s a, R s a + h (next s a) - h s ≤ θ)
    (n : ℕ) (s : ℕ → S) (a : ℕ → Bool) (htraj : ∀ i, s (i + 1) = next (s i) (a i)) :
    (∑ i ∈ range n, R (s i) (a i)) ≤ n * θ + h (s 0) - h (s n) := by
  induction n with
  | zero => simp
  | succ m ih =>
      rw [Finset.sum_range_succ]
      have hstep : R (s m) (a m) + h (s (m + 1)) - h (s m) ≤ θ := by
        rw [htraj m]; exact hcert _ _
      push_cast
      linarith

/-- Per-symbol form: with `|h| ≤ B`, every trajectory averages at most `θ + 2B/n`. -/
theorem cert_rate {S : Type*} (next : S → Bool → S) (R : S → Bool → ℝ)
    (h : S → ℝ) (θ B : ℝ) (hcert : ∀ s a, R s a + h (next s a) - h s ≤ θ)
    (hB : ∀ s, |h s| ≤ B)
    (n : ℕ) (s : ℕ → S) (a : ℕ → Bool) (htraj : ∀ i, s (i + 1) = next (s i) (a i)) :
    (∑ i ∈ range n, R (s i) (a i)) ≤ n * θ + 2 * B := by
  have h1 := cert_bound next R h θ hcert n s a htraj
  have h2 : h (s 0) ≤ B := (abs_le.mp (hB (s 0))).2
  have h3 : -B ≤ h (s n) := (abs_le.mp (hB (s n))).1
  linarith

/-! ### Step 7 — the truncation tail, charged in the right direction -/

/-- `(1 - r) * ∑_{j<J} r^j = 1 - r^J`. -/
theorem geom_sum_mul (r : ℝ) : ∀ J : ℕ, (1 - r) * (∑ j ∈ range J, r ^ j) = 1 - r ^ J := by
  intro J
  induction J with
  | zero => simp
  | succ m ih => rw [Finset.sum_range_succ]; ring_nf; ring_nf at ih; linarith

/-- For `0 ≤ r < 1`, every partial geometric sum is at most `1 / (1 - r)`. -/
theorem geom_sum_le_inv (r : ℝ) (hr0 : 0 ≤ r) (hr1 : r < 1) (J : ℕ) :
    (∑ j ∈ range J, r ^ j) ≤ 1 / (1 - r) := by
  have hpos : (0:ℝ) < 1 - r := by linarith
  rw [le_div_iff₀ hpos]
  have := geom_sum_mul r J
  have hnn : (0:ℝ) ≤ r ^ J := pow_nonneg hr0 J
  nlinarith [this, hnn]

/-- **Geometric remainder.**  If the terms contract by `r < 1` beyond index `M`, every
finite piece of the tail is at most `t M * r / (1 - r)`.  Adding this to a finite sum turns
a lower estimate of the tail into an upper bound on it. -/
theorem geom_tail (t : ℕ → ℝ) (r : ℝ) (M : ℕ) (hr0 : 0 ≤ r) (hr1 : r < 1)
    (ht : ∀ i, 0 ≤ t i) (hstep : ∀ i, M ≤ i → t (i + 1) ≤ r * t i) (J : ℕ) :
    (∑ j ∈ range J, t (M + 1 + j)) ≤ t M * (r / (1 - r)) := by
  have hpos : (0:ℝ) < 1 - r := by linarith
  have htM : 0 ≤ t M := ht M
  have key : ∀ j, t (M + 1 + j) ≤ r ^ (j + 1) * t M := by
    intro j
    induction j with
    | zero =>
        have := hstep M le_rfl
        simpa using this
    | succ m ih =>
        have h1 : t (M + 1 + m + 1) ≤ r * t (M + 1 + m) :=
          hstep (M + 1 + m) (by omega)
        have h2 : r * t (M + 1 + m) ≤ r * (r ^ (m + 1) * t M) :=
          mul_le_mul_of_nonneg_left ih hr0
        calc t (M + 1 + (m + 1)) = t (M + 1 + m + 1) := by ring_nf
          _ ≤ r * t (M + 1 + m) := h1
          _ ≤ r * (r ^ (m + 1) * t M) := h2
          _ = r ^ (m + 1 + 1) * t M := by ring
  calc (∑ j ∈ range J, t (M + 1 + j))
      ≤ ∑ j ∈ range J, r ^ (j + 1) * t M := Finset.sum_le_sum (fun j _ => key j)
    _ = r * t M * (∑ j ∈ range J, r ^ j) := by
        rw [Finset.mul_sum]; apply Finset.sum_congr rfl; intro j _; ring
    _ ≤ r * t M * (1 / (1 - r)) := by
        have := geom_sum_le_inv r hr0 hr1 J
        have hrt : 0 ≤ r * t M := mul_nonneg hr0 htM
        exact mul_le_mul_of_nonneg_left this hrt
    _ = t M * (r / (1 - r)) := by ring

/-! ### Steps 1–9 assembled -/

/-- **Assembly.**  Each ingredient enters as its own hypothesis, in the direction in which
it is actually available, so a truncation charged the wrong way cannot typecheck.

`Hexact`  : Steps 1–3 (duality + the exact identity `H(Y|x) = n H_b(d) − E[log emb]`).
`Hwindow` : Steps 4–5 (the window bound on `E[log emb]`), charged UP.
`Htrunc`  : Steps 6–7 (the gap truncation), the discarded mass ADDED as `n * tb`.
`Hcert`   : Steps 8–9 (the certificate of `cert_rate`).

Mutation-tested: `linarith` fails if the tail is omitted from the reward, or if the window
bound is flipped, or if the tail is subtracted rather than added. -/
theorem assembly (n : ℕ)
    (Cn Hb Elogemb ElogQ psiSum rewSum tb θ boundary : ℝ)
    (_htb    : 0 ≤ tb)   -- unused by the correct assembly; needed for the mutation tests
    (Hexact  : Cn ≤ -(n : ℝ) * Hb + Elogemb + ElogQ)
    (Hwindow : Elogemb ≤ psiSum)
    (Htrunc  : ElogQ ≤ rewSum + (n : ℝ) * tb)
    (Hcert   : psiSum + rewSum + (n : ℝ) * tb ≤ (n : ℝ) * θ + boundary) :
    Cn ≤ (n : ℝ) * (θ - Hb) + boundary := by
  linarith

/-! ### Step 10 — extension in `d` -/

/-- **Extension.**  Given the Rahmati–Duman monotonicity of `C(d)/(1−d)` (hypothesis
`hmono`, cited from Pinto–Ribeiro 2026 Lemma 11), a bound at `d` becomes a bound at every
larger `e`. -/
theorem extend_in_d (C : ℝ → ℝ) (d v : ℝ) (hd1 : d < 1)
    (hmono : ∀ e, d ≤ e → e < 1 → C e / (1 - e) ≤ C d / (1 - d))
    (hC : C d ≤ v) :
    ∀ e, d ≤ e → e < 1 → C e ≤ v * (1 - e) / (1 - d) := by
  intro e hde he1
  have h1 : (0:ℝ) < 1 - d := by linarith
  have h2 : (0:ℝ) < 1 - e := by linarith
  have hne1 : (1:ℝ) - d ≠ 0 := ne_of_gt h1
  have hne2 : (1:ℝ) - e ≠ 0 := ne_of_gt h2
  have hm := hmono e hde he1
  have hpos : (0:ℝ) < (1 - e) * (1 - d) := mul_pos h2 h1
  have key := mul_le_mul_of_nonneg_right hm hpos.le
  have e1 : C e / (1 - e) * ((1 - e) * (1 - d)) = C e * (1 - d) := by field_simp
  have e2 : C d / (1 - d) * ((1 - e) * (1 - d)) = C d * (1 - e) := by field_simp
  rw [e1, e2] at key
  rw [le_div_iff₀ h1]
  nlinarith [key, hC, h2]

end BDC
