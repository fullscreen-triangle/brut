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

const OpportunityCard = ({ icon, title, description, needs, delay }) => (
  <motion.div
    className="glass-card glow-border"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <div className="text-3xl mb-3">{icon}</div>
    <h3 className="text-light font-bold text-lg mb-2">{title}</h3>
    <p className="text-lightMuted text-sm leading-relaxed mb-4">{description}</p>
    <div className="space-y-2">
      {needs.map((need, i) => (
        <div key={i} className="flex items-start gap-2">
          <span className="text-primary text-xs mt-1 flex-shrink-0">&#9670;</span>
          <span className="text-lightMuted text-sm">{need}</span>
        </div>
      ))}
    </div>
  </motion.div>
);

const IPCard = ({ title, description, status, market, delay }) => (
  <motion.div
    className="glass-card glow-border"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <div className="flex items-center justify-between mb-3">
      <h3 className="text-light font-semibold">{title}</h3>
      <span className="text-xs font-mono px-2 py-0.5 rounded bg-accent/20 text-accentLight">
        {status}
      </span>
    </div>
    <p className="text-lightMuted text-sm leading-relaxed mb-3">{description}</p>
    <div className="text-xs text-lightMuted font-mono bg-darkAlt/60 rounded p-2">
      Target: {market}
    </div>
  </motion.div>
);

const TimelineItem = ({ phase, title, description, delay }) => (
  <motion.div
    className="flex gap-4"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <div className="flex flex-col items-center">
      <div className="w-10 h-10 rounded-full bg-primary/20 border border-primary/40 flex items-center justify-center text-primary font-mono text-xs font-bold">
        {phase}
      </div>
      <div className="w-px h-full bg-primary/20 mt-2" />
    </div>
    <div className="pb-8">
      <h3 className="text-light font-semibold mb-1">{title}</h3>
      <p className="text-lightMuted text-sm leading-relaxed">{description}</p>
    </div>
  </motion.div>
);

export default function Collaborate() {
  return (
    <>
      <Head>
        <title>Collaborate — BRUT Framework</title>
      </Head>
      <TransitionEffect />

      {/* Header */}
      <Layout className="!pt-16 !pb-8">
        <div className="text-center max-w-3xl mx-auto">
          <AnimatedText
            text="Collaborate With Us"
            className="!text-6xl xl:!text-5xl md:!text-3xl"
          />
          <motion.p
            className="text-lightMuted text-lg mt-4 leading-relaxed md:text-base"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
          >
            The BRUT framework opens new research directions and commercial
            opportunities. We are seeking academic collaborators, clinical
            partners, and strategic investors.
          </motion.p>
        </div>
      </Layout>

      {/* For Researchers */}
      <Layout className="!py-8">
        <motion.div
          className="text-center mb-10"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2 className="section-heading">For Researchers</h2>
          <p className="section-subheading mx-auto">
            Open problems and collaboration opportunities across multiple
            disciplines.
          </p>
        </motion.div>

        <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
          <OpportunityCard
            icon="&#9829;"
            title="Clinical Cardiology"
            delay={0}
            description="Validate the cardiac regime classification framework against prospective clinical data. The CHF paradox (Theorem 11) predicts two distinct failure modes distinguishable by (R_c, S_e) coordinates."
            needs={[
              "Access to ICU continuous monitoring data",
              "Prospective arrhythmia cohort studies",
              "Decoherence cascade detection experiments",
              "Cardiac resynchronisation therapy outcome prediction",
            ]}
          />
          <OpportunityCard
            icon="&#9733;"
            title="Computational Neuroscience"
            delay={1}
            description="Test the cardiac-neural coupling law and consciousness window formula with paired EEG + ECG recordings. The framework predicts specific R_n/R_c ratios per cognitive state."
            needs={[
              "Simultaneous EEG + ECG during cognitive tasks",
              "Sleep polysomnography with high-resolution ECG",
              "Anaesthesia depth monitoring studies",
              "Temporal binding window measurements",
            ]}
          />
          <OpportunityCard
            icon="&#9650;"
            title="Exercise & Altitude Physiology"
            delay={2}
            description="Validate the O2-partition coupling constant and altitude degradation curves. The framework predicts specific R_n degradation profiles as a function of arterial PaO2."
            needs={[
              "Altitude chamber studies with concurrent HRV + cognitive testing",
              "VO2max protocols with partition regime tracking",
              "Recovery dynamics in high-altitude acclimatisation",
              "Hypoxia-induced decoherence measurements",
            ]}
          />
          <OpportunityCard
            icon="&#9670;"
            title="Wearable Sensor Engineering"
            delay={3}
            description="Implement and validate the novel composite metrics (PCHR, TCC, CSCI) on commercial wearable platforms. The sensor disambiguation framework transforms single-sensor readings into multi-dimensional state estimates."
            needs={[
              "Multi-sensor wearable platforms (PPG + temp + accel + EDA)",
              "Real-time embedded implementation of R_c estimator",
              "Clinical validation of PCHR vs standard HR monitoring",
              "Longitudinal cohort data for CSCI normative ranges",
            ]}
          />
          <OpportunityCard
            icon="&#9711;"
            title="Mathematical Physics"
            delay={4}
            description="Extend the partition formalism to new domains. The C(n) = 2n² axiom may apply beyond physiology — any bounded system with hierarchical state space structure."
            needs={[
              "Rigorous proof of uniqueness of quadratic capacity",
              "Extension to non-equilibrium statistical mechanics",
              "Connection to quantum information geometry",
              "Category-theoretic formalisation of partition hierarchy",
            ]}
          />
          <OpportunityCard
            icon="&#9678;"
            title="Metabolic & Endocrine Systems"
            delay={5}
            description="Apply the partition framework to glucose regulation, hormonal cycling, and metabolic syndrome. The S-entropy coordinates should capture metabolic health as a trajectory in (S_k, S_t, S_e) space."
            needs={[
              "Continuous glucose monitor + HRV paired datasets",
              "Cortisol circadian rhythm with concurrent cardiac data",
              "Metabolic syndrome cohort studies",
              "Insulin sensitivity as a function of partition regime",
            ]}
          />
        </div>
      </Layout>

      {/* For Investors */}
      <div className="border-y border-primary/10">
        <Layout className="!py-12">
          <motion.div
            className="text-center mb-10"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            <h2 className="section-heading">For Investors</h2>
            <p className="section-subheading mx-auto">
              The sensor disambiguation IP transforms commodity wearable hardware
              into clinically meaningful diagnostic tools.
            </p>
          </motion.div>

          <div className="grid grid-cols-2 gap-6 lg:grid-cols-1">
            <IPCard
              title="Partition-Coupled Heart Rate (PCHR)"
              status="Patent-Ready"
              delay={0}
              description="Decomposes observed HR into intrinsic, metabolic, and autonomic components using partition equations. Same sensor hardware, fundamentally richer clinical information. HR=90 bpm becomes actionable: exercise recovery vs fever vs anxiety vs altitude."
              market="Wearable health platforms, clinical monitoring, sports performance"
            />
            <IPCard
              title="S-Entropy Health Coordinates"
              status="Patent-Ready"
              delay={1}
              description="Maps multi-sensor wearable data into a 3D health state (S_k, S_t, S_e). Replaces 15+ disconnected metrics with a single navigable coordinate system. Enables trajectory-based health monitoring: not where you are, but where you're heading."
              market="Digital health platforms, preventive medicine, insurance risk scoring"
            />
            <IPCard
              title="Temperature-Corrected Coherence (TCC)"
              status="Validated"
              delay={2}
              description="Removes metabolic bias from cardiac coherence estimates using Arrhenius correction with skin temperature. Reduces longitudinal tracking noise by 12%, improving trend detection for chronic disease monitoring."
              market="Remote patient monitoring, cardiology telehealth, clinical trials"
            />
            <IPCard
              title="Cross-Scale Coherence Index (CSCI)"
              status="Validated"
              delay={3}
              description="Detects inter-system decoupling from wrist sensors alone. First demonstration that REM cardiac-neural decoupling is detectable without EEG. Opens sleep staging and neurological screening from a wristband."
              market="Sleep technology, neurology screening, mental health monitoring"
            />
          </div>
        </Layout>
      </div>

      {/* Roadmap */}
      <Layout className="!py-12">
        <motion.div
          className="text-center mb-10"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2 className="section-heading">Commercialisation Roadmap</h2>
        </motion.div>

        <div className="max-w-2xl mx-auto">
          <TimelineItem
            phase="1"
            title="Academic Validation"
            description="Complete peer-reviewed publication of all four papers. Prospective clinical validation of regime classification and coupling formula with hospital partners. Target: 3 independent replication studies."
            delay={0}
          />
          <TimelineItem
            phase="2"
            title="IP Protection & Licensing"
            description="Patent filing for PCHR, CSCI, and S-entropy coordinate mapping algorithms. Provisional patents on sensor disambiguation methods. Establish licensing framework for wearable OEMs."
            delay={1}
          />
          <TimelineItem
            phase="3"
            title="SDK & Platform Development"
            description="Build a real-time SDK implementing R_c estimation, PCHR decomposition, and CSCI computation. Target integration with Apple HealthKit, Google Health Connect, and Oura API."
            delay={2}
          />
          <TimelineItem
            phase="4"
            title="Clinical Product"
            description="Partner with a wearable manufacturer to launch the first partition-aware health monitoring product. Regulatory pathway: FDA 510(k) for cardiac coherence monitoring, CE marking for EU market."
            delay={3}
          />
        </div>
      </Layout>

      {/* Contact */}
      <div className="border-t border-primary/10">
        <Layout className="!py-16">
          <div className="text-center max-w-2xl mx-auto">
            <motion.h2
              className="section-heading"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              Get in Touch
            </motion.h2>
            <motion.p
              className="text-lightMuted mb-6"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
            >
              Whether you are a researcher interested in collaboration, a clinician
              with relevant datasets, or an investor exploring health technology
              opportunities, we would like to hear from you.
            </motion.p>
            <motion.div
              className="flex flex-col items-center gap-3"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4 }}
            >
              <div className="text-light font-semibold">Kundai Sachikonye</div>
              <div className="text-lightMuted text-sm">
                Technical University of Munich
              </div>
              <div className="text-lightMuted text-sm">
                Department of Mathematical Physiology
              </div>
            </motion.div>
          </div>
        </Layout>
      </div>
    </>
  );
}
