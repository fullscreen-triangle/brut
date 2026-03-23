import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import Head from "next/head";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.12, duration: 0.6 },
  }),
};

const DerivationStep = ({ number, title, from, to, equations, color, delay }) => (
  <motion.div
    className="glass-card glow-border relative"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <div className="flex items-start gap-4">
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-full ${color} flex items-center justify-center text-white font-bold text-sm`}
      >
        {number}
      </div>
      <div className="flex-1">
        <h3 className="text-light font-bold text-lg mb-1">{title}</h3>
        <div className="text-lightMuted text-xs font-mono mb-3">
          {from} &rarr; {to}
        </div>
        <div className="space-y-2">
          {equations.map((eq, i) => (
            <div key={i} className="equation-block !text-sm !p-2">
              {eq}
            </div>
          ))}
        </div>
      </div>
    </div>
  </motion.div>
);

const RegimeCard = ({ name, range, examples, color, delay }) => (
  <motion.div
    className="glass-card glow-border"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <div className="flex items-center gap-3 mb-2">
      <div className={`w-3 h-3 rounded-full ${color}`} />
      <h3 className="text-light font-semibold">{name}</h3>
      <span className="text-lightMuted text-xs font-mono ml-auto">{range}</span>
    </div>
    <ul className="space-y-1">
      {examples.map((ex, i) => (
        <li key={i} className="text-lightMuted text-sm flex items-start gap-2">
          <span className="text-primary text-xs mt-1">&#9670;</span>
          {ex}
        </li>
      ))}
    </ul>
  </motion.div>
);

const CoordinateAxis = ({ symbol, name, description, maps, delay }) => (
  <motion.div
    className="glass-card glow-border text-center"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <div className="text-4xl font-bold gradient-text mb-2 font-mono">{symbol}</div>
    <h3 className="text-light font-semibold mb-1">{name}</h3>
    <p className="text-lightMuted text-sm mb-3">{description}</p>
    <div className="text-xs text-lightMuted font-mono bg-darkAlt/60 rounded p-2">
      {maps}
    </div>
  </motion.div>
);

export default function Framework() {
  return (
    <>
      <Head>
        <title>Framework — BRUT</title>
      </Head>
      <TransitionEffect />

      {/* Header */}
      <Layout className="!pt-16 !pb-8">
        <div className="text-center max-w-3xl mx-auto">
          <AnimatedText
            text="The Framework"
            className="!text-6xl xl:!text-5xl md:!text-3xl"
          />
          <motion.p
            className="text-lightMuted text-lg mt-4 leading-relaxed md:text-base"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
          >
            All physiological systems — cardiovascular, neural, metabolic — emerge
            from a single axiom about how bounded systems partition their state
            space.
          </motion.p>
        </div>
      </Layout>

      {/* The Axiom */}
      <div className="border-y border-primary/10">
        <Layout className="!py-12">
          <div className="max-w-3xl mx-auto text-center">
            <motion.div
              className="text-primary font-mono text-sm tracking-widest uppercase mb-4"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
            >
              The Single Axiom
            </motion.div>
            <motion.div
              className="equation-block !text-3xl !p-8 mb-6 md:!text-xl"
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
            >
              C(n) = 2n&sup2;
            </motion.div>
            <motion.p
              className="text-lightMuted leading-relaxed"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
            >
              The partition capacity function C(n) counts the number of
              distinguishable categorical states at depth level n. This quadratic
              form arises from spherical symmetry constraints in bounded phase space,
              yielding the sequence (2, 8, 18, 32, 50, ...). From this single
              function, we derive all subsequent structures: entropy, coherence,
              coupling, and dynamics.
            </motion.p>
          </div>
        </Layout>
      </div>

      {/* S-Entropy Coordinates */}
      <Layout className="!py-12">
        <motion.div
          className="text-center mb-10"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2 className="section-heading">S-Entropy Coordinates</h2>
          <p className="section-subheading mx-auto">
            Every physiological state maps to a point in the S-entropy cube (S&#x2096;,
            S&#x209C;, S&#x2091;) &#x2208; [0,1]&sup3;.
          </p>
        </motion.div>

        <div className="grid grid-cols-3 gap-6 lg:grid-cols-1">
          <CoordinateAxis
            symbol="S&#x2096;"
            name="Knowledge Depth"
            description="How much of the partition hierarchy is being explored. High S_k indicates deep, complex state space traversal."
            maps="HRV complexity, EEG spectral richness"
            delay={0}
          />
          <CoordinateAxis
            symbol="S&#x209C;"
            name="Temporal Integration"
            description="Position in the circadian-ultradian temporal hierarchy. Encodes time-scale coherence and phase relationships."
            maps="Circadian phase, autocorrelation decay"
            delay={1}
          />
          <CoordinateAxis
            symbol="S&#x2091;"
            name="Entropy Utilisation"
            description="Fraction of available partition capacity being used. Low S_e indicates either rigid lock-in or active decoupling."
            maps="HRV ratio to maximum, regime occupancy"
            delay={2}
          />
        </div>
      </Layout>

      {/* Partition Regimes */}
      <div className="border-y border-primary/10">
        <Layout className="!py-12">
          <motion.div
            className="text-center mb-10"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            <h2 className="section-heading">Partition Regimes</h2>
            <p className="section-subheading mx-auto">
              The Kuramoto order parameter R classifies physiological states into
              five regimes, each with distinct dynamics and clinical meaning.
            </p>
          </motion.div>

          <div className="grid grid-cols-5 gap-4 xl:grid-cols-3 lg:grid-cols-2 md:grid-cols-1">
            <RegimeCard
              name="Phase-Locked"
              range="R > 0.95"
              color="bg-blue-600"
              delay={0}
              examples={[
                "Deep sleep cardiac",
                "Pathological rigidity (CHF)",
                "Loss of complexity",
              ]}
            />
            <RegimeCard
              name="Coherent"
              range="0.80 - 0.95"
              color="bg-blue-400"
              delay={1}
              examples={[
                "Normal sinus rhythm",
                "Awake resting state",
                "Coupled oscillators",
              ]}
            />
            <RegimeCard
              name="Cascade"
              range="0.50 - 0.80"
              color="bg-green-500"
              delay={2}
              examples={[
                "Light sleep transitions",
                "Exercise recovery",
                "Moderate variability",
              ]}
            />
            <RegimeCard
              name="Aperture"
              range="0.30 - 0.50"
              color="bg-amber-500"
              delay={3}
              examples={[
                "Ventricular tachycardia",
                "High autonomic flux",
                "Transitional states",
              ]}
            />
            <RegimeCard
              name="Turbulent"
              range="R < 0.30"
              color="bg-red-500"
              delay={4}
              examples={[
                "Atrial fibrillation (R=0.170)",
                "Bigeminy (R=0.018)",
                "Maximal desynchronisation",
              ]}
            />
          </div>
        </Layout>
      </div>

      {/* Derivation Chain */}
      <Layout className="!py-12">
        <motion.div
          className="text-center mb-10"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2 className="section-heading">Derivation Chain</h2>
          <p className="section-subheading mx-auto">
            Starting from C(n) = 2n&sup2;, each physiological law is derived — not
            assumed — through a chain of mathematical consequences.
          </p>
        </motion.div>

        <div className="grid grid-cols-2 gap-6 lg:grid-cols-1">
          <DerivationStep
            number={1}
            title="Partition Entropy"
            from="C(n) = 2n&sup2;"
            to="S-entropy coordinates"
            color="bg-primary"
            delay={0}
            equations={[
              "S = k_B ln C(n) = k_B ln(2n\u00B2)",
              "(S_k, S_t, S_e) \u2208 [0,1]\u00B3",
            ]}
          />
          <DerivationStep
            number={2}
            title="Kuramoto Order Parameter"
            from="Phase distribution on S\u00B9"
            to="Coherence regimes"
            color="bg-primary"
            delay={1}
            equations={[
              "R = |N\u207B\u00B9 \u03A3 exp(i\u03B8_j)|",
              "R_c = exp(-2\u03C0\u00B2\u00B7CV\u00B2)",
            ]}
          />
          <DerivationStep
            number={3}
            title="Cardiac Equations of State"
            from="Partition boundary conditions"
            to="Pressure-volume thermodynamics"
            color="bg-emerald"
            delay={2}
            equations={[
              "PdV + VdP = C(n)kT",
              "E_{es} = \u2202P/\u2202V |_{S,n}",
            ]}
          />
          <DerivationStep
            number={4}
            title="Frank-Starling & Windkessel"
            from="PV equation of state"
            to="Hemodynamic laws"
            color="bg-emerald"
            delay={3}
            equations={[
              "SV = SV_max \u00B7 (1 - e^{-k\u00B7V_ed})",
              "P(t) = P_d \u00B7 e^{-t/RC}",
            ]}
          />
          <DerivationStep
            number={5}
            title="Cardiac-Neural Coupling"
            from="Cross-scale coherence"
            to="Universal coupling law"
            color="bg-accent"
            delay={4}
            equations={[
              "R_n/R_c = 0.87/\u221AR_c",
              "\u0394t_C = T/(2\u03C0\u221A(R_c\u00B7R_n))",
            ]}
          />
          <DerivationStep
            number={6}
            title="Metabolic Integration"
            from="O\u2082-partition coupling"
            to="Temperature-corrected coherence"
            color="bg-rose"
            delay={5}
            equations={[
              "\u03BA_{O\u2082} = 4.7\u00D710\u207B\u00B3 s\u207B\u00B9",
              "TCC = R_c \u00B7 exp[(E_a/k_B)(1/T - 1/T\u2080)]",
            ]}
          />
        </div>
      </Layout>

      {/* Key Discoveries */}
      <div className="border-t border-primary/10">
        <Layout className="!py-12">
          <motion.div
            className="text-center mb-10"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            <h2 className="section-heading">Key Discoveries</h2>
            <p className="section-subheading mx-auto">
              Predictions confirmed and revised through empirical validation against
              PhysioNet databases and 86 nights of wearable sensor data.
            </p>
          </motion.div>

          <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
            {[
              {
                title: "CHF Paradox Resolved",
                text: "Congestive heart failure shows HIGHER R_c (0.797) than normal sinus rhythm (0.710) — pathological phase-locking, not loss of coherence. Distinguished by low entropy utilisation S_e (Theorem 11: Two Failure Modes).",
                tag: "CONFIRMED",
                tagColor: "bg-emerald/20 text-emerald",
              },
              {
                title: "REM Active Decoupling",
                text: "During REM sleep, the cardiac-neural gap reaches 0.375 — the largest of any stage. The cardiac system maintains coherent delivery while the neural system explores turbulent-to-cascade states (Corollary 8).",
                tag: "DISCOVERED",
                tagColor: "bg-primary/20 text-primaryDark",
              },
              {
                title: "Light Sleep Highest RMSSD",
                text: "Light sleep (N2) exhibits the highest RMSSD (65.8 ms), exceeding REM (61.0 ms) and Deep (51.8 ms). Attributable to K-complex and spindle-driven episodic autonomic bursts.",
                tag: "NEW FINDING",
                tagColor: "bg-accent/20 text-accentLight",
              },
              {
                title: "Bigeminy Reclassified",
                text: "Initially predicted as aperture regime. Empirical R_c = 0.018 (deep turbulent) — the alternating N-V-N-V pattern maximally anti-correlates successive RR intervals, collapsing coherence below atrial fibrillation.",
                tag: "REVISED",
                tagColor: "bg-rose/20 text-rose",
              },
              {
                title: "AFIB Regime Confirmed",
                text: "Atrial fibrillation R_c = 0.170, firmly turbulent. 78.8% epoch classification accuracy. Cohen's d = 33.2 vs normal sinus rhythm. The strongest single validation of the regime boundary framework.",
                tag: "CONFIRMED",
                tagColor: "bg-emerald/20 text-emerald",
              },
              {
                title: "Coupling Formula Validated",
                text: "The cardiac-neural coupling formula R_n/R_c = 0.87/sqrt(R_c) shows best fit during N1/N2 sleep (error = 0.011) and breaks down during REM (error = 0.308), exactly as predicted.",
                tag: "CONFIRMED",
                tagColor: "bg-emerald/20 text-emerald",
              },
            ].map((item, i) => (
              <motion.div
                key={i}
                className="glass-card glow-border"
                custom={i}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
              >
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-light font-semibold text-sm">{item.title}</h3>
                  <span
                    className={`text-xs font-mono px-2 py-0.5 rounded ${item.tagColor}`}
                  >
                    {item.tag}
                  </span>
                </div>
                <p className="text-lightMuted text-sm leading-relaxed">{item.text}</p>
              </motion.div>
            ))}
          </div>
        </Layout>
      </div>
    </>
  );
}
