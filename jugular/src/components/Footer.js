import Link from "next/link";
import React from "react";
import Layout from "./Layout";

const Footer = () => {
  return (
    <footer className="w-full border-t border-primary/10 font-medium text-sm text-lightMuted">
      <Layout className="!py-8 flex items-center justify-between lg:flex-col lg:gap-2">
        <span>{new Date().getFullYear()} &copy; BRUT Framework. All rights reserved.</span>
        <div className="flex items-center gap-6">
          <Link href="/framework" className="hover:text-light transition-colors">
            Framework
          </Link>
          <Link href="/research" className="hover:text-light transition-colors">
            Research
          </Link>
          <Link href="/collaborate" className="hover:text-light transition-colors">
            Collaborate
          </Link>
        </div>
        <span>Technical University of Munich</span>
      </Layout>
    </footer>
  );
};

export default Footer;
