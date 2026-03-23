import { motion } from "framer-motion";
import Link from "next/link";
import React from "react";

let MotionLink = motion(Link);

const Logo = () => {
  return (
    <div className="flex flex-col items-center justify-center mt-2">
      <MotionLink
        href="/"
        className="flex items-center justify-center rounded-full w-16 h-16
        bg-gradient-to-br from-primary to-primary/60 text-white
        text-lg font-bold tracking-wider border-2 border-primary/30"
        whileHover={{
          boxShadow: [
            "0 0 10px rgba(59,130,246,0.3)",
            "0 0 25px rgba(59,130,246,0.5)",
            "0 0 10px rgba(59,130,246,0.3)",
          ],
          transition: { duration: 1.5, repeat: Infinity },
        }}
      >
        BR
      </MotionLink>
    </div>
  );
};

export default Logo;
