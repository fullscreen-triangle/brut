import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import Head from "next/head";
import Link from "next/link";
import dynamic from "next/dynamic";

const GLBViewer = dynamic(
  () => import("@/components/models/GLBViewer").then((mod) => mod.GLBViewer),
  { ssr: false }
);

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.15, duration: 0.6 },
  }),
};

const StatCard = ({ value, label, delay }) => (
  <motion.div
    className="glass-card text-center glow-border"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <div className="text-3xl font-bold gradient-text mb-1">{value}</div>
    <div className="text-lightMuted text-sm">{label}</div>
  </motion.div>
);

const EquationCard = ({ title, equation, description, delay }) => (
  <motion.div
    className="glass-card glow-border"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <h3 className="text-primary font-semibold text-sm uppercase tracking-wider mb-3">
      {title}
    </h3>
    <div className="equation-block mb-3 text-lg">{equation}</div>
    <p className="text-lightMuted text-sm leading-relaxed">{description}</p>
  </motion.div>
);

const PillarCard = ({ number, title, items, color, delay }) => (
  <motion.div
    className="glass-card glow-border relative overflow-hidden"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <div
      className={`absolute top-0 left-0 w-1 h-full ${color}`}
    />
    <div className="pl-4">
      <div className="text-lightMuted text-xs font-mono mb-1">0{number}</div>
      <h3 className="text-light font-bold text-lg mb-3">{title}</h3>
      <ul className="space-y-1.5">
        {items.map((item, i) => (
          <li key={i} className="text-lightMuted text-sm flex items-start gap-2">
            <span className="text-primary mt-1 text-xs">&#9670;</span>
            {item}
          </li>
        ))}
      </ul>
    </div>
  </motion.div>
);

export default function Home() {
  return (
    <>
      <Head>
        <title>BRUT Framework — Unified Physiological Mathematics</title>
      </Head>
      <TransitionEffect />

      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-50" />
        <div className="absolute inset-0 bg-radial-glow" />

        <Layout className="!pt-16 !pb-8 relative">
          <div className="flex items-center gap-12 max-w-6xl mx-auto lg:flex-col">
            {/* Left: Text */}
            <div className="flex-1 lg:text-center">
              <motion.div
                className="mb-4 text-primary font-mono text-sm tracking-widest uppercase"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
              >
                A Single Axiom. All of Physiology.
              </motion.div>

              <AnimatedText
                text="The BRUT Framework"
                className="!text-7xl !text-left xl:!text-5xl lg:!text-4xl lg:!text-center md:!text-3xl"
              />

              <motion.p
                className="text-lightMuted text-xl mt-4 leading-relaxed md:text-base sm:text-sm"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2, duration: 0.6 }}
              >
                Deriving cardiovascular dynamics, neural coherence, and metabolic
                coupling from a single partition capacity axiom:{" "}
                <span className="text-primaryDark font-mono">C(n) = 2n&sup2;</span>
              </motion.p>

              <motion.div
                className="flex gap-4 mt-8 md:flex-col md:w-full lg:justify-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.5, duration: 0.6 }}
              >
                <Link
                  href="/framework"
                  className="px-8 py-3 bg-primary hover:bg-primary/80 text-white font-semibold rounded-lg transition-all duration-200 shadow-glow hover:shadow-glow-lg"
                >
                  Explore the Framework
                </Link>
                <Link
                  href="/research"
                  className="px-8 py-3 border border-primary/30 hover:border-primary/60 text-light font-semibold rounded-lg transition-all duration-200"
                >
                  View Research
                </Link>
              </motion.div>
            </div>

            {/* Right: 3D Heart Model */}
            <motion.div
              className="flex-1 flex items-center justify-center lg:w-full"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.8, duration: 1 }}
            >
              <div className="w-[420px] h-[420px] xl:w-[340px] xl:h-[340px] md:w-[280px] md:h-[280px] relative">
                <div className="absolute inset-0 rounded-full bg-primary/5 animate-pulse-slow" />
                <GLBViewer
                  modelPath="/model/beating-heart.glb"
                  autoRotate={true}
                  rotationSpeed={0.4}
                  modelScale={2.5}
                  modelPosition={[0, -0.5, 0]}
                  cameraPosition={[0, 0, 5]}
                  cameraFov={45}
                  enableOrbit={true}
                  ambientIntensity={0.7}
                  pointLightIntensity={1.2}
                  playAnimation={true}
                  className="w-full h-full"
                />
              </div>
            </motion.div>
          </div>
        </Layout>
      </div>

      {/* Core Equations */}
      <Layout className="!pt-8 !pb-12">
        <motion.div
          className="text-center mb-10"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2 className="section-heading">Core Equations</h2>
          <p className="section-subheading mx-auto">
            From one axiom, a complete mathematical physiology emerges.
          </p>
        </motion.div>

        <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
          <EquationCard
            title="Partition Capacity"
            equation="C(n) = 2n&sup2;"
            description="The foundational axiom. The number of distinguishable categorical states at partition depth n follows a quadratic scaling law derived from spherical symmetry constraints."
            delay={0}
          />
          <EquationCard
            title="S-Entropy"
            equation="S = k&#x2095; ln C(n)"
            description="Entropy over the partition hierarchy. Connects information-theoretic capacity to thermodynamic entropy via the three coordinates (S_k, S_t, S_e)."
            delay={1}
          />
          <EquationCard
            title="Kuramoto Coherence"
            equation="R&#x2093; = exp(-2&pi;&sup2;&middot;CV&sup2;)"
            description="Cardiac coherence estimated from heart rate variability. CV = RMSSD&middot;HR/60000 maps beat-to-beat variation to the Kuramoto order parameter."
            delay={2}
          />
          <EquationCard
            title="Frank-Starling Law"
            equation="SV = SV&#x2098;&#x2090;&#x2093; &middot; (1 - e&#x207B;&#x1D4F;&#x22C5;&#x1D62;&#x1D65;)"
            description="Stroke volume as a function of preload, derived from partition boundary conditions on the pressure-volume loop."
            delay={3}
          />
          <EquationCard
            title="Cardiac-Neural Coupling"
            equation="R&#x2099;/R&#x2093; = 0.87/&radic;R&#x2093;"
            description="The ratio of neural to cardiac coherence follows a universal scaling law, valid during coupled states. Breaks down during REM sleep."
            delay={4}
          />
          <EquationCard
            title="Consciousness Window"
            equation="&Delta;t&#x2082; = T / (2&pi;&radic;(R&#x2093;&middot;R&#x2099;))"
            description="The temporal integration window for conscious experience, derived from the geometric mean of cardiac and neural coherence."
            delay={5}
          />
        </div>
      </Layout>

      {/* Stats */}
      <div className="border-y border-primary/10">
        <Layout className="!py-12">
          <div className="grid grid-cols-4 gap-6 lg:grid-cols-2 sm:grid-cols-1">
            <StatCard value="1" label="Foundational Axiom" delay={0} />
            <StatCard value="5" label="Derived Subsystems" delay={1} />
            <StatCard value="17+" label="Testable Predictions" delay={2} />
            <StatCard value="4" label="Published Papers" delay={3} />
          </div>
        </Layout>
      </div>

      {/* Derivation Pillars */}
      <Layout className="!py-12">
        <motion.div
          className="text-center mb-10"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2 className="section-heading">Derivation Hierarchy</h2>
          <p className="section-subheading mx-auto">
            Each physiological system is derived from the same partition axiom,
            forming a unified hierarchy from cardiac mechanics to conscious experience.
          </p>
        </motion.div>

        <div className="grid grid-cols-2 gap-6 lg:grid-cols-1">
          <PillarCard
            number={1}
            title="Cardiovascular Mechanics"
            color="bg-primary"
            delay={0}
            items={[
              "Pressure-volume loops from partition boundary conditions",
              "Frank-Starling, Windkessel, and baroreflex as emergent properties",
              "Cardiac equations of state: PdV + VdP = C(n)kT formalism",
              "Disease classification via Kuramoto regime boundaries",
            ]}
          />
          <PillarCard
            number={2}
            title="Neural Coherence"
            color="bg-emerald"
            delay={1}
            items={[
              "EEG band structure from partition depth selection",
              "Consciousness as temporal integration over coherence window",
              "Sleep architecture as regime traversal sequence",
              "REM active decoupling: cardiac-neural gap = 0.375",
            ]}
          />
          <PillarCard
            number={3}
            title="Metabolic Coupling"
            color="bg-accent"
            delay={2}
            items={[
              "Oxygen transport as partition-level coupling constant",
              "Temperature-dependent coherence via Arrhenius formalism",
              "Metabolic cost of cognitive geometries (thought metabolism)",
              "Altitude degradation curves from O2-partition coupling",
            ]}
          />
          <PillarCard
            number={4}
            title="Sensor Disambiguation"
            color="bg-rose"
            delay={3}
            items={[
              "Partition-Coupled Heart Rate (PCHR) decomposition",
              "S-entropy health coordinates from wearable sensors",
              "Cross-Scale Coherence Index for inter-system coupling",
              "Temperature-corrected coherence removing metabolic bias",
            ]}
          />
        </div>
      </Layout>

      {/* CTA */}
      <div className="border-t border-primary/10">
        <Layout className="!py-16">
          <div className="text-center max-w-2xl mx-auto">
            <motion.h2
              className="section-heading"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              Join the Research
            </motion.h2>
            <motion.p
              className="section-subheading mx-auto mb-8"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
            >
              We are seeking collaborators with expertise in clinical cardiology,
              computational neuroscience, and wearable sensor engineering.
              Investment opportunities available for sensor disambiguation IP.
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4 }}
            >
              <Link
                href="/collaborate"
                className="px-8 py-3 bg-primary hover:bg-primary/80 text-white font-semibold rounded-lg transition-all duration-200 shadow-glow hover:shadow-glow-lg"
              >
                Collaborate With Us
              </Link>
            </motion.div>
          </div>
        </Layout>
      </div>
    </>
  );
}
