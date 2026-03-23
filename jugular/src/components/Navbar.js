import Link from "next/link";
import React, { useState } from "react";
import Logo from "./Logo";
import { useRouter } from "next/router";
import { GithubIcon } from "./Icons";
import { motion } from "framer-motion";

const NavLink = ({ href, title, className = "" }) => {
  const router = useRouter();
  const isActive = router.asPath === href;

  return (
    <Link
      href={href}
      className={`${className} relative group text-light/80 hover:text-light transition-colors duration-200`}
    >
      {title}
      <span
        className={`inline-block h-[2px] bg-primary absolute left-0 -bottom-1
          group-hover:w-full transition-[width] ease duration-300
          ${isActive ? "w-full" : "w-0"}`}
      >
        &nbsp;
      </span>
    </Link>
  );
};

const MobileNavLink = ({ href, title, className = "", toggle }) => {
  const router = useRouter();

  const handleClick = () => {
    toggle();
    router.push(href);
  };

  return (
    <button
      className={`${className} relative group text-light/80 hover:text-light transition-colors`}
      onClick={handleClick}
    >
      {title}
      <span
        className={`inline-block h-[2px] bg-primary absolute left-0 -bottom-1
          group-hover:w-full transition-[width] ease duration-300
          ${router.asPath === href ? "w-full" : "w-0"}`}
      >
        &nbsp;
      </span>
    </button>
  );
};

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const handleClick = () => {
    setIsOpen(!isOpen);
  };

  return (
    <header
      className="w-full flex items-center justify-between px-32 py-6 font-medium z-10 text-light
      lg:px-16 relative md:px-12 sm:px-8 border-b border-primary/10"
    >
      {/* Mobile hamburger */}
      <button
        type="button"
        className="flex-col items-center justify-center hidden lg:flex"
        aria-controls="mobile-menu"
        aria-expanded={isOpen}
        onClick={handleClick}
      >
        <span className="sr-only">Open main menu</span>
        <span
          className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${
            isOpen ? "rotate-45 translate-y-1" : "-translate-y-0.5"
          }`}
        />
        <span
          className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${
            isOpen ? "opacity-0" : "opacity-100"
          } my-0.5`}
        />
        <span
          className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${
            isOpen ? "-rotate-45 -translate-y-1" : "translate-y-0.5"
          }`}
        />
      </button>

      {/* Desktop nav */}
      <div className="w-full flex justify-between items-center lg:hidden">
        <nav className="flex items-center justify-center gap-6">
          <NavLink href="/" title="Home" />
          <NavLink href="/framework" title="Framework" />
          <NavLink href="/research" title="Research" />
          <NavLink href="/collaborate" title="Collaborate" />
        </nav>
        <nav className="flex items-center justify-center gap-4">
          <motion.a
            target="_blank"
            className="w-6 text-light/60 hover:text-light transition-colors"
            href="https://github.com"
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.9 }}
            aria-label="GitHub repository"
          >
            <GithubIcon />
          </motion.a>
        </nav>
      </div>

      {/* Mobile menu */}
      {isOpen ? (
        <motion.div
          className="min-w-[70vw] sm:min-w-[90vw] flex justify-between items-center flex-col fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
          py-32 bg-darkAlt/95 rounded-lg z-50 backdrop-blur-md border border-primary/20"
          initial={{ scale: 0, x: "-50%", y: "-50%", opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
        >
          <nav className="flex items-center justify-center flex-col gap-4">
            <MobileNavLink toggle={handleClick} href="/" title="Home" />
            <MobileNavLink toggle={handleClick} href="/framework" title="Framework" />
            <MobileNavLink toggle={handleClick} href="/research" title="Research" />
            <MobileNavLink toggle={handleClick} href="/collaborate" title="Collaborate" />
          </nav>
          <nav className="flex items-center justify-center mt-6">
            <motion.a
              target="_blank"
              className="w-6 text-light/60 hover:text-light"
              href="https://github.com"
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.9 }}
              aria-label="GitHub repository"
            >
              <GithubIcon />
            </motion.a>
          </nav>
        </motion.div>
      ) : null}

      <div className="absolute left-[50%] top-2 translate-x-[-50%]">
        <Logo />
      </div>
    </header>
  );
};

export default Navbar;
