import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: [
    "@prisma/client",
    "prisma",
    "@prisma/engines",
    "@prisma/client-runtime-utils",
  ],
};

export default nextConfig;
