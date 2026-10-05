# Choi Covariance, Single-Fermion Insertions, and Effective Transfer Matrices

This file is a reconstructed markdown write-up of the technical chatlog. It preserves the main sequence of questions, answers, formulae, corrections, and final conclusions about U(1)-symmetric fermionic Choi covariances, postselected creation/loss operators, and effective transfer matrices.

---

## 0. Notation

We work with a U(1)-symmetric fermionic Gaussian Choi state on doubled single-particle Hilbert space

\[
\mathcal H_L\oplus \mathcal H_R.
\]

The complex covariance is

\[
\Sigma=
\begin{pmatrix}
\Sigma_{LL} & \Sigma_{LR}\\
\Sigma_{RL} & \Sigma_{RR}
\end{pmatrix},
\qquad \Sigma^2=1
\]

for a pure Gaussian Choi state. The ordinary correlation matrix is

\[
C=\frac{1+\Sigma}{2}
=
\begin{pmatrix}
C_{LL} & C_{LR}\\
C_{RL} & C_{RR}
\end{pmatrix},
\qquad C^2=C.
\]

A quasiparticle mode is written as

\[
\chi^\dagger=\sum_i \chi_i^* c_i^\dagger,
\qquad
|\chi\rangle=\sum_i \chi_i |i\rangle,
\qquad
P_\chi=|\chi\rangle\langle \chi|.
\]

For a right-leg insertion, use the doubled vector

\[
u_R=
\begin{pmatrix}
0\\ |\chi\rangle
\end{pmatrix}_{L\oplus R}.
\]

---

## 1. First issue: does purity still apply after a single fermion insertion?

### Question

If one applies a single fermion creation or annihilation operator to a Choi state, can one still impose the purity condition?

### Answer

Yes. For a branch-resolved postselected operator,

\[
|K\rangle\rangle \mapsto \chi_R^\dagger |K\rangle\rangle
\]

or

\[
|K\rangle\rangle \mapsto \chi_R |K\rangle\rangle,
\]

the normalized state remains pure and Gaussian, provided the branch amplitude is nonzero. Therefore its covariance still satisfies

\[
\Sigma_\pm^2=1.
\]

What fails is not purity. What fails is the use of a full-rank even transfer-matrix coordinate chart when the postselected odd state develops null/disconnected directions.

---

## 2. Why naive regulation of the final odd covariance can fail

A gain-type covariance for one reset mode can look like

\[
\Sigma_{\rm gain}=
\begin{pmatrix}
P & Q\\
Q & P
\end{pmatrix},
\qquad Q=1-P.
\]

This is pure:

\[
\Sigma_{\rm gain}^2=1.
\]

But a proposed softened regulator such as

\[
\Sigma_\epsilon=
\begin{pmatrix}
P+\epsilon Q & Q+\epsilon P\\
Q+\epsilon P & P+\epsilon Q
\end{pmatrix}
\]

is generally **not** pure for finite \(\epsilon\). On the \(P\)-sector,

\[
\Sigma_\epsilon|_P=
\begin{pmatrix}
1 & \epsilon\\
\epsilon & 1
\end{pmatrix},
\]

so

\[
\Sigma_\epsilon^2|_P=
\begin{pmatrix}
1+\epsilon^2 & 2\epsilon\\
2\epsilon & 1+\epsilon^2
\end{pmatrix}
\neq 1.
\]

Therefore block identities derived using purity no longer have to hold for such a regulator. This explains why two transfer extraction formulae can disagree under a bad regulator.

The correct regularization should be applied at the level of the parent operator/transfer matrix or by using the exact rank-one Gaussian update, not by arbitrarily interpolating the final odd covariance.

---

## 3. Pure Choi representation of a single fermion operator

Use the reference state for one mode

\[
|\Omega\rangle_{LR}
=
\frac{1}{\sqrt2}
\left(|0_L1_R\rangle+|1_L0_R\rangle\right).
\]

Then

\[
\chi_R^\dagger |\Omega\rangle
\propto |1_L1_R\rangle,
\]

and

\[
\chi_R |\Omega\rangle
\propto |0_L0_R\rangle.
\]

Thus the pure Choi representations are

\[
\chi^\dagger
\quad\longleftrightarrow\quad
|1_L1_R\rangle,
\]

\[
\chi
\quad\longleftrightarrow\quad
|0_L0_R\rangle.
\]

For many modes with

\[
P=P_\chi,
\qquad Q=1-P,
\]

the corresponding covariances are

\[
\Sigma_{\chi^\dagger}
=
\begin{pmatrix}
P & Q\\
Q & P
\end{pmatrix},
\]

and

\[
\Sigma_\chi
=
\begin{pmatrix}
-P & Q\\
Q & -P
\end{pmatrix}.
\]

These are perfectly valid pure Gaussian Choi covariances.

The issue is not that odd operators lack Choi states. The issue is that an odd operator does not define an ordinary invertible even Gaussian transfer matrix by conjugation.

---

## 4. Single fermion operator times an even Gaussian operator

Consider

\[
K=\chi^\dagger V,
\]

where \(V\) is an even Gaussian operator, possibly nonunitary. Its Choi state is

\[
|K\rangle\rangle
=
\chi_R^\dagger V_R |\Omega\rangle.
\]

If

\[
|V\rangle\rangle=V_R|\Omega\rangle
\]

has covariance \(\Sigma_V\) and correlation projector

\[
C_V=\frac{1+\Sigma_V}{2},
\]

then creation by a generic doubled mode \(u\) updates the covariance by filling the empty projection of \(u\):

\[
C_+
=
C_V+
\frac{(1-C_V)uu^\dagger(1-C_V)}{u^\dagger(1-C_V)u}.
\]

Annihilation removes the occupied projection:

\[
C_-
=
C_V-
\frac{C_Vuu^\dagger C_V}{u^\dagger C_V u}.
\]

For a right-leg insertion,

\[
u=u_R=
\begin{pmatrix}0\\ |\chi\rangle\end{pmatrix}.
\]

---

## 5. Generic covariance update under creation/loss

### 5.1 In \(\Sigma\)-language

For creation on the right leg,

\[
|K\rangle\rangle \mapsto \chi_R^\dagger |K\rangle\rangle,
\]

define

\[
D_+=u_R^\dagger(1-\Sigma)u_R
=\langle \chi |(1-\Sigma_{RR})|\chi\rangle
=\operatorname{Tr}[(1-\Sigma_{RR})P_\chi].
\]

If \(D_+\neq 0\), then

\[
\Sigma_+
=
\Sigma+
\frac{(1-\Sigma)u_Ru_R^\dagger(1-\Sigma)}{u_R^\dagger(1-\Sigma)u_R}.
\]

Blockwise:

\[
\boxed{
\Sigma_{LL}^{(+)}
=
\Sigma_{LL}
+
\frac{\Sigma_{LR}P_\chi\Sigma_{RL}}{D_+}
}
\]

\[
\boxed{
\Sigma_{LR}^{(+)}
=
\Sigma_{LR}
-
\frac{\Sigma_{LR}P_\chi(1-\Sigma_{RR})}{D_+}
}
\]

\[
\boxed{
\Sigma_{RL}^{(+)}
=
\Sigma_{RL}
-
\frac{(1-\Sigma_{RR})P_\chi\Sigma_{RL}}{D_+}
}
\]

\[
\boxed{
\Sigma_{RR}^{(+)}
=
\Sigma_{RR}
+
\frac{(1-\Sigma_{RR})P_\chi(1-\Sigma_{RR})}{D_+}
}
\]

For annihilation/loss on the right leg,

\[
|K\rangle\rangle \mapsto \chi_R |K\rangle\rangle,
\]

define

\[
D_-=u_R^\dagger(1+\Sigma)u_R
=\langle \chi |(1+\Sigma_{RR})|\chi\rangle
=\operatorname{Tr}[(1+\Sigma_{RR})P_\chi].
\]

If \(D_-\neq0\), then

\[
\Sigma_-
=
\Sigma-
\frac{(1+\Sigma)u_Ru_R^\dagger(1+\Sigma)}{u_R^\dagger(1+\Sigma)u_R}.
\]

Blockwise:

\[
\boxed{
\Sigma_{LL}^{(-)}
=
\Sigma_{LL}
-
\frac{\Sigma_{LR}P_\chi\Sigma_{RL}}{D_-}
}
\]

\[
\boxed{
\Sigma_{LR}^{(-)}
=
\Sigma_{LR}
-
\frac{\Sigma_{LR}P_\chi(1+\Sigma_{RR})}{D_-}
}
\]

\[
\boxed{
\Sigma_{RL}^{(-)}
=
\Sigma_{RL}
-
\frac{(1+\Sigma_{RR})P_\chi\Sigma_{RL}}{D_-}
}
\]

\[
\boxed{
\Sigma_{RR}^{(-)}
=
\Sigma_{RR}
-
\frac{(1+\Sigma_{RR})P_\chi(1+\Sigma_{RR})}{D_-}
}
\]

---

### 5.2 In \(C\)-language

Let

\[
C=\frac{1+\Sigma}{2}.
\]

For creation,

\[
C_+
=
C+
\frac{(1-C)u_Ru_R^\dagger(1-C)}{u_R^\dagger(1-C)u_R}.
\]

Define

\[
d_+=\langle \chi |(1-C_{RR})|\chi\rangle
=\operatorname{Tr}[(1-C_{RR})P_\chi].
\]

Then

\[
\boxed{
C_{LL}^{(+)}
=
C_{LL}
+
\frac{C_{LR}P_\chi C_{RL}}{d_+}
}
\]

\[
\boxed{
C_{LR}^{(+)}
=
C_{LR}
-
\frac{C_{LR}P_\chi(1-C_{RR})}{d_+}
}
\]

\[
\boxed{
C_{RL}^{(+)}
=
C_{RL}
-
\frac{(1-C_{RR})P_\chi C_{RL}}{d_+}
}
\]

\[
\boxed{
C_{RR}^{(+)}
=
C_{RR}
+
\frac{(1-C_{RR})P_\chi(1-C_{RR})}{d_+}
}
\]

For annihilation,

\[
C_-
=
C-
\frac{Cu_Ru_R^\dagger C}{u_R^\dagger C u_R}.
\]

Define

\[
d_-=
\langle \chi |C_{RR}|\chi\rangle
=\operatorname{Tr}[C_{RR}P_\chi].
\]

Then

\[
\boxed{
C_{LL}^{(-)}
=
C_{LL}
-
\frac{C_{LR}P_\chi C_{RL}}{d_-}
}
\]

\[
\boxed{
C_{LR}^{(-)}
=
C_{LR}
-
\frac{C_{LR}P_\chi C_{RR}}{d_-}
}
\]

\[
\boxed{
C_{RL}^{(-)}
=
C_{RL}
-
\frac{C_{RR}P_\chi C_{RL}}{d_-}
}
\]

\[
\boxed{
C_{RR}^{(-)}
=
C_{RR}
-
\frac{C_{RR}P_\chi C_{RR}}{d_-}
}
\]

The denominators obey

\[
D_+=2d_+,
\qquad
D_-=2d_-.
\]

In \(C\)-language the denominators \(d_\pm\) are the branch probabilities.

---

## 6. Relation to set/reset maps

The postselected creation branch

\[
\rho\mapsto \chi^\dagger\rho\chi
\]

is not the same as the deterministic reset-to-filled channel.

The deterministic reset-to-filled operation is

\[
\mathcal R_+(\rho)=N\rho N+\chi^\dagger\rho\chi,
\qquad N=\chi^\dagger\chi.
\]

At covariance level this acts as

\[
C\mapsto QCQ+P,
\qquad Q=1-P.
\]

Similarly, reset-to-empty is

\[
\mathcal R_-(\rho)=(1-N)\rho(1-N)+\chi\rho\chi^\dagger,
\]

and acts as

\[
C\mapsto QCQ.
\]

If the occupation \(N_\chi\) has already been measured and the outcome is kept, then the branch-resolved measurement plus feedback preserves purity. If the measurement outcome is forgotten, the result is generally mixed.

---

## 7. Even transfer matrix reconstruction formula

For an even full-rank transfer matrix

\[
T\equiv T_p,
\]

the Choi correlation blocks are

\[
\boxed{
C_{LL}=(1+TT^\dagger)^{-1}
}
\]

\[
\boxed{
C_{LR}=(1+TT^\dagger)^{-1}T
}
\]

\[
\boxed{
C_{RL}=T^\dagger(1+TT^\dagger)^{-1}
}
\]

\[
\boxed{
C_{RR}=T^\dagger(1+TT^\dagger)^{-1}T
}
\]

Equivalently define

\[
L=(1+TT^\dagger)^{-1},
\qquad
R=(1+T^\dagger T)^{-1}.
\]

Then

\[
C_{LL}=L,
\qquad
C_{LR}=LT=TR,
\qquad
C_{RL}=T^\dagger L=RT^\dagger,
\qquad
C_{RR}=1-R.
\]

The covariance is

\[
\Sigma(T)=2C(T)-1.
\]

In the full-rank even chart, the transfer matrix can be extracted as

\[
\boxed{
T=C_{LL}^{-1}C_{LR}
}
\]

or equivalently

\[
\boxed{
T=(1+\Sigma_{LL})^{-1}\Sigma_{LR}
}
\]

with the convention used here.

---

## 8. Effective transfer matrix after right-leg creation/loss

Starting from a pre-insertion even covariance determined by \(T\), the effective particle transfer matrix extracted from the postselected Choi covariance is

\[
T_{\rm eff}^{(\pm)}=(C_{LL}^{(\pm)})^{-1}C_{LR}^{(\pm)}.
\]

With

\[
P=P_\chi,
\qquad Q=1-P,
\]

the regulated/pseudoinverse result is

\[
\boxed{
T_{\rm eff}^{(+)}=TQ
}
\]

for creation and

\[
\boxed{
T_{\rm eff}^{(-)}=TQ
}
\]

for annihilation/loss.

Thus a right-leg single-fermion insertion removes the \(\chi\)-column direction:

\[
\boxed{
T_p\mapsto T_p(1-P_\chi).
}
\]

If the insertion were on the left leg, it would instead remove the corresponding row direction:

\[
T_p\mapsto (1-P_\chi)T_p.
\]

---

## 9. Does \(T_{\rm eff}\) fully characterize the post-creation covariance?

No.

This is the main subtlety.

The effective transfer matrix characterizes the homogeneous LR propagation sector. It does not characterize the occupation of the null/disconnected direction.

For example, take

\[
T=1.
\]

Then the reference Choi state is

\[
|\Omega\rangle
=\frac{|0_L1_R\rangle+|1_L0_R\rangle}{\sqrt2}.
\]

After creation,

\[
\chi_R^\dagger|\Omega\rangle\propto |1_L1_R\rangle,
\]

so

\[
C_+=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}.
\]

After annihilation/loss,

\[
\chi_R|\Omega\rangle\propto |0_L0_R\rangle,
\]

so

\[
C_-=
\begin{pmatrix}
0&0\\
0&0
\end{pmatrix}.
\]

In both cases the LR bridge is gone, so

\[
T_{\rm eff}=0.
\]

But the two Choi covariances are different. Therefore \(T_{\rm eff}\) does not fully characterize the postselected covariance.

---

## 10. What goes wrong if one reconstructs from \(T_{\rm eff}\) using the even formula?

If one uses the even formula with

\[
T_{\rm eff}=Q,
\]

then on the null \(P\)-sector one obtains

\[
C_{\rm even}(Q)|_P
=
\begin{pmatrix}
1&0\\
0&0
\end{pmatrix},
\]

which corresponds to

\[
|1_L0_R\rangle.
\]

But the actual post-creation Choi state on that sector is

\[
|1_L1_R\rangle,
\]

with

\[
C_+|_P
=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}.
\]

Likewise, the actual post-loss state is

\[
|0_L0_R\rangle,
\]

with

\[
C_-|_P
=
\begin{pmatrix}
0&0\\
0&0
\end{pmatrix}.
\]

Therefore

\[
\boxed{
\Sigma_{\rm post}\neq \Sigma(T_{\rm eff})
\quad \text{in general.}
}
\]

Rather,

\[
\boxed{
\Sigma_{\rm post}=\Sigma(T_{\rm eff},\text{null-sector occupation data}).
}
\]

For creation, the missing datum is

\[
P_\chi C_{RR}^{(+)}P_\chi=P_\chi.
\]

For loss, the missing datum is

\[
P_\chi C_{RR}^{(-)}P_\chi=0.
\]

---

## 11. Can one just regularize \(T_{\rm eff}\)?

One can regularize it, but regularizing \(T_{\rm eff}\) alone does not uniquely recover the postselected odd covariance.

For example, suppose

\[
T_{\rm eff}=0.
\]

An even regularization

\[
T_\epsilon=\epsilon
\]

gives

\[
C(T_\epsilon)
=
\begin{pmatrix}
\frac{1}{1+\epsilon^2} & \frac{\epsilon}{1+\epsilon^2}\\[0.4em]
\frac{\epsilon}{1+\epsilon^2} & \frac{\epsilon^2}{1+\epsilon^2}
\end{pmatrix}
\to
\begin{pmatrix}
1&0\\
0&0
\end{pmatrix}.
\]

This is the even null completion \(|1_L0_R\rangle\), not the post-creation state \(|1_L1_R\rangle\) and not the post-loss state \(|0_L0_R\rangle\).

Another even regularization

\[
T_\epsilon=\epsilon^{-1}
\]

gives

\[
C(T_\epsilon)
\to
\begin{pmatrix}
0&0\\
0&1
\end{pmatrix},
\]

which is the other even null completion \(|0_L1_R\rangle\).

Thus even transfer-matrix regularizations give opposite-sign null completions:

\[
|1_L0_R\rangle,
\qquad
|0_L1_R\rangle.
\]

Odd postselected insertions give same-sign null completions:

\[
\chi^\dagger:
\quad |1_L1_R\rangle,
\]

\[
\chi:
\quad |0_L0_R\rangle.
\]

Therefore:

\[
\boxed{
T_{\rm eff}\text{ can be regularized, but its regularization is not enough.}
}
\]

The missing datum is the null-sector occupation/branch.

---

## 12. Correct final picture

The correct hierarchy is:

1. **Pure Choi covariance for an individual Kraus amplitude**

   A postselected single fermion insertion has a valid pure Gaussian Choi covariance satisfying

   \[
   \Sigma^2=1.
   \]

2. **Rank-one update is the correct covariance transformation**

   For creation:

   \[
   C_+=C+\frac{(1-C)u_Ru_R^\dagger(1-C)}{u_R^\dagger(1-C)u_R}.
   \]

   For loss:

   \[
   C_-=C-\frac{Cu_Ru_R^\dagger C}{u_R^\dagger C u_R}.
   \]

3. **Effective transfer matrix after right-leg insertion**

   \[
   T_{\rm eff}=T(1-P_\chi).
   \]

   This captures the non-null LR propagation sector.

4. **Transfer matrix alone misses null-sector occupation**

   The data

   \[
   T_{\rm eff}P_\chi=0
   \]

   only says that the \(\chi\)-column no longer propagates. It does not say whether the disconnected right-leg mode is empty or filled.

5. **Creation and loss differ precisely in the missing null-sector datum**

   Creation:

   \[
   P_\chi C_{RR}P_\chi=P_\chi.
   \]

   Loss:

   \[
   P_\chi C_{RR}P_\chi=0.
   \]

6. **Thus the full singular transfer data is more like**

   \[
   \boxed{
   (T_{\rm eff},\;P_{\rm null},\;n_{\rm null})
   }
   \]

   where

   \[
   n_{\rm null}=1
   \]

   for postselected creation and

   \[
   n_{\rm null}=0
   \]

   for postselected loss.

---

## 13. Final distilled statement

The final conclusion of the chatlog is:

\[
\boxed{
\text{The postselected single-fermion Choi state is pure and Gaussian.}
}
\]

\[
\boxed{
\text{Its covariance is obtained by a rank-one creation/loss update.}
}
\]

\[
\boxed{
\text{The effective transfer matrix is }T_{\rm eff}=T(1-P_\chi)
\text{ for right-leg insertion.}
}
\]

\[
\boxed{
\text{But }T_{\rm eff}\text{ does not fully characterize the postselected covariance.}
}
\]

\[
\boxed{
\text{It misses the null-sector occupation: filled for }\chi^\dagger,
\text{ empty for }\chi.
}
\]

This resolves the apparent contradiction: purity is preserved, Choi representation is valid, and the transfer matrix is still meaningful, but the singular effective transfer matrix is not a complete coordinate chart for the odd postselected Choi covariance.

