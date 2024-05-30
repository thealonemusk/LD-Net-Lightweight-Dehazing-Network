import Image from "next/image";
import Link from "next/link";

export default function LandingPage() {
    return (
      <>
        {/* Page Container */}
        <div
          id="page-container"
          className="mx-auto flex min-h-dvh w-full min-w-[320px] flex-col bg-gray-100 dark:bg-gray-900 dark:text-gray-100"
        >
          {/* Page Content */}
          <main id="page-content" className="flex max-w-full flex-auto flex-col">
            {/* Hero */}
            <div className="relative overflow-hidden bg-white dark:bg-gray-900 dark:text-gray-100">
              {/* Main Header */}
              <header
                id="page-header"
                className="relative flex flex-none items-center py-8"
              >
                {/* Main Header Content */}
                <div className="container mx-auto flex flex-col gap-4 px-4 text-center md:flex-row md:items-center md:justify-between md:gap-0 lg:px-8 xl:max-w-7xl">
                  <div>
                    <a
                      href="#"
                      className="group inline-flex items-center gap-2 text-lg font-bold tracking-wide text-gray-900 hover:text-gray-600 dark:text-gray-100 dark:hover:text-gray-300"
                    >
                      <svg
                        className="hi-mini hi-cube-transparent inline-block size-5 text-teal-600 transition group-hover:scale-110 dark:text-teal-400"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                        aria-hidden="true"
                      >
                        <path
                          fillRule="evenodd"
                          d="M9.638 1.093a.75.75 0 01.724 0l2 1.104a.75.75 0 11-.724 1.313L10 2.607l-1.638.903a.75.75 0 11-.724-1.313l2-1.104zM5.403 4.287a.75.75 0 01-.295 1.019l-.805.444.805.444a.75.75 0 01-.724 1.314L3.5 7.02v.73a.75.75 0 01-1.5 0v-2a.75.75 0 01.388-.657l1.996-1.1a.75.75 0 011.019.294zm9.194 0a.75.75 0 011.02-.295l1.995 1.101A.75.75 0 0118 5.75v2a.75.75 0 01-1.5 0v-.73l-.884.488a.75.75 0 11-.724-1.314l.806-.444-.806-.444a.75.75 0 01-.295-1.02zM7.343 8.284a.75.75 0 011.02-.294L10 8.893l1.638-.903a.75.75 0 11.724 1.313l-1.612.89v1.557a.75.75 0 01-1.5 0v-1.557l-1.612-.89a.75.75 0 01-.295-1.019zM2.75 11.5a.75.75 0 01.75.75v1.557l1.608.887a.75.75 0 01-.724 1.314l-1.996-1.101A.75.75 0 012 14.25v-2a.75.75 0 01.75-.75zm14.5 0a.75.75 0 01.75.75v2a.75.75 0 01-.388.657l-1.996 1.1a.75.75 0 11-.724-1.313l1.608-.887V12.25a.75.75 0 01.75-.75zm-7.25 4a.75.75 0 01.75.75v.73l.888-.49a.75.75 0 01.724 1.313l-2 1.104a.75.75 0 01-.724 0l-2-1.104a.75.75 0 11.724-1.313l.888.49v-.73a.75.75 0 01.75-.75z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span>Clearzz</span>
                    </a>
                  </div>
                  <div className="flex flex-col gap-4 text-center md:flex-row md:items-center md:justify-between md:gap-0">
                    <nav className="space-x-3 md:space-x-6">
                      <a
                        href="#"
                        className="text-sm font-semibold text-gray-900 hover:text-teal-600 dark:text-gray-100 dark:hover:text-teal-400"
                      >
                        <span>Features</span>
                      </a>
                      <a
                        href="#"
                        className="text-sm font-semibold text-gray-900 hover:text-teal-600 dark:text-gray-100 dark:hover:text-teal-400"
                      >
                        <span>Pricing</span>
                      </a>
                      <a
                        href="#"
                        className="text-sm font-semibold text-gray-900 hover:text-teal-600 dark:text-gray-100 dark:hover:text-teal-400"
                      >
                        <span>Support</span>
                      </a>
                    </nav>
                    <div className="mx-6 hidden h-8 w-px bg-gradient-to-b from-transparent via-gray-300 to-transparent dark:via-gray-700 md:block" />
                    <div className="flex items-center justify-center gap-2">
                      <Link
                        href="../dashboard"
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm font-semibold leading-5 text-gray-800 hover:border-gray-300 hover:text-gray-900 hover:shadow-sm focus:ring focus:ring-gray-300/25 active:border-gray-200 active:shadow-none dark:border-gray-700 dark:bg-transparent dark:text-gray-300 dark:hover:border-gray-600 dark:hover:text-gray-200 dark:focus:ring-gray-600/40 dark:active:border-gray-700"
                      >
                        <span>Dashboard</span>
                        <svg
                          className="hi-mini hi-arrow-right inline-block size-5 opacity-50"
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                          aria-hidden="true"
                        >
                          <path
                            fillRule="evenodd"
                            d="M3 10a.75.75 0 01.75-.75h10.638L10.23 5.29a.75.75 0 111.04-1.08l5.5 5.25a.75.75 0 010 1.08l-5.5 5.25a.75.75 0 11-1.04-1.08l4.158-3.96H3.75A.75.75 0 013 10z"
                            clipRule="evenodd"
                          />
                        </svg>
                      </Link>
                    </div>
                  </div>
                </div>
                {/* END Main Header Content */}
              </header>
              {/* END Main Header */}
  
              {/* Hero Content */}
              <div className="container relative mx-auto px-4 py-16 lg:px-8 lg:py-32 xl:max-w-7xl">
                <div className="text-center">
                  <div className="mb-2 inline-flex rounded border border-gray-200 bg-gray-100 px-2 py-1 text-sm font-medium leading-4 text-gray-800 dark:border-gray-600/50 dark:bg-gray-700/50 dark:text-gray-200">
                    dehaze image processing
                  </div>
                  <h1 className="mb-4 text-4xl font-black text-black dark:text-white">
                    Clear the Fog
                    <span className="text-teal-600 dark:text-teal-500">
                       from your images
                    </span>
                  </h1>
                  <h2 className="mx-auto text-xl font-medium leading-relaxed text-gray-700 dark:text-gray-300 lg:w-2/3">
                    Remove fog from images using deep learning. Upload your image and see the magic.
                  </h2>
                </div>
                <div className="flex flex-col gap-2 pb-16 pt-10 sm:flex-row sm:items-center sm:justify-center sm:gap-3">
                  <Link
                    href="#"
                    className="inline-flex items-center justify-center gap-2 rounded-lg border border-teal-700 bg-teal-700 px-7 py-3.5 font-semibold leading-6 text-white hover:border-teal-600 hover:bg-teal-600 hover:text-white focus:ring focus:ring-teal-400/50 active:border-teal-700 active:bg-teal-700 dark:focus:ring-teal-400/90"
                  >
                    <span>Get Started</span>
                  </Link>
                  <a
                    href="#"
                    className="inline-flex items-center justify-center gap-2 rounded-lg border border-gray-200 bg-white px-7 py-3.5 font-semibold leading-6 text-gray-800 hover:border-gray-300 hover:text-gray-900 hover:shadow-sm focus:ring focus:ring-gray-300/25 active:border-gray-200 active:shadow-none dark:border-gray-700 dark:bg-transparent dark:text-gray-300 dark:hover:border-gray-600 dark:hover:text-gray-200 dark:focus:ring-gray-600/40 dark:active:border-gray-700"
                  >
                    <span>Learn more</span>
                  </a>
                </div>
              </div>
              {/* END Hero Content */}
            </div>
            {/* END Hero */}
  
            {/* Section */}
            <div className="bg-gray-100 dark:bg-gray-800/50 dark:text-gray-100">
              <div className="container mx-auto px-4 py-16 lg:px-8 lg:py-32 xl:max-w-7xl">
                <div className="grid grid-cols-1 gap-12 md:grid-cols-2 lg:gap-16">
                  <div className="flex flex-col gap-6">
                    <h2 className="text-3xl font-black text-black dark:text-white">
                      What is Clearzz?
                    </h2>
                    <p className="text-lg text-gray-700 dark:text-gray-300">
                      Clearzz is a simple and easy-to-use tool that removes fog from images. It is a great tool for photographers, designers, and anyone who wants to improve the quality of their images.
                    </p>
                  </div>
                  <div className="flex flex-col gap-6">
                    <h2 className="text-3xl font-black text-black dark:text-white">
                      Why Clearzz?
                    </h2>
                    <p className="text-lg text-gray-700 dark:text-gray-300">
                      Clearzz uses deep learning to remove fog from images. It is a simple process that involves uploading an image and waiting for the magic to happen.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            {/* END Section */}
  
            {/* Section */}
            <div className="bg-white dark:bg-gray-800 dark:text-gray-100">
              <div className="container mx-auto px-4 py-16 lg:px-8 lg:py-32 xl:max-w-7xl">
                <div className="grid grid-cols-1 gap-12 md:grid-cols-2 lg:gap-16">
                    <div className="flex flex-col gap-6 p-10">
                      <h2 className="text-3xl font-black text-black dark:text-white">
                        How it works
                      </h2>
                      <p className="text-lg text-gray-700 dark:text-gray-300">
                        Clearzz uses a deep learning model to remove fog from images. The model is trained on a large dataset of foggy and clear images, and it is able to learn the patterns of fog and remove it from images. The process is simple and easy to use, and it can be done in just a few seconds.
                      </p>
                    </div>
                    <div className="flex flex-col gap-6">
                      <div className="relative">
                        <Image width={320 } height={320} src="/test3.jpeg"  alt="Before" className="absolute h-auto max-w-md md:max-w-md rounded-md" />
                        <div className="absolute top-0 left-0 bg-slate-400 text-white text-xs px-2 py-1 rounded-sm">Before</div>
                      </div>
                      <div className="relative">
                        <Image  width={320 } height={320} src="/dehazed_image.png" alt="After" className="absolute top-10 left-40 h-auto max-w-md md:max-w-md rounded-md" />
                        <div className="absolute top-10 left-40 bg-slate-400 text-white text-xs px-2 py-1 rounded-sm">After</div>
                      </div>
                    </div>
                </div> 
              </div>
            </div>
            {/* END Section */}
  
            {/* Footer */}
            <footer
              id="page-footer"
              className="bg-white dark:bg-gray-900 dark:text-gray-100"
            >
              <div className="container mx-auto px-4 py-16 lg:px-8 lg:py-32 xl:max-w-7xl">
                <div className="grid grid-cols-1 gap-12 md:grid-cols-3 md:gap-6 lg:gap-10">
                  <div className="space-y-6">
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-400/75">
                      Products
                    </h4>
                    <nav className="flex flex-col gap-3 text-sm">
                      <a
                        href="#"
                        className="font-medium text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-gray-50"
                      >
                        Solutions
                      </a>
                      <a
                        href="#"
                        className="font-medium text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-gray-50"
                      >
                        Features
                      </a>
                      <a
                        href="#"
                        className="font-medium text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-gray-50"
                      >
                        Pricing Plans
                      </a>
                      <a
                        href="#"
                        className="font-medium text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-gray-50"
                      >
                        Analytics
                      </a>
                      <a
                        href="#"
                        className="font-medium text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-gray-50"
                      >
                        Support Center
                      </a>
                    </nav>
                  </div>
                  <div className="space-y-6">
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-400/75">
                      Legal
                    </h4>
                    <nav className="flex flex-col gap-3 text-sm">
                      <a
                        href="#"
                        className="font-medium text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-gray-50"
                      >
                        Team
                      </a>
                      <a
                        href="#"
                        className="font-medium text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-gray-50"
                      >
                        Terms of Service
                      </a>
                      <a
                        href="#"
                        className="font-medium text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-gray-50"
                      >
                        Privacy Policy
                      </a>
                      <a
                        href="#"
                        className="font-medium text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-gray-50"
                      >
                        Cookies
                      </a>
                      <a
                        href="#"
                        className="font-medium text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-gray-50"
                      >
                        Refunds
                      </a>
                    </nav>
                  </div>
                  <div className="space-y-6">
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-400/75">
                    Clearzz Inc
                    </h4>
                    <div className="text-sm leading-relaxed">
                      1080 Sunshine Valley, Suite 2563
                      <br />
                      San Francisco, CA 85214
                    </div>
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-400/75">
                      Join Our Newsletter
                    </h4>
                    <form className="space-y-3 sm:flex sm:items-center sm:gap-2 sm:space-y-0">
                      <div className="flex items-center">
                        <input
                          type="email"
                          id="email"
                          name="email"
                          placeholder="Enter your email"
                          className="block w-full grow rounded-l-lg border border-gray-200 px-3 py-2 text-sm leading-5 placeholder-gray-500 focus:z-1 focus:border-teal-500 focus:ring focus:ring-teal-500/50 dark:border-gray-600 dark:bg-gray-900 dark:placeholder-gray-400 dark:focus:border-teal-500"
                        />
                        <button
                          type="submit"
                          className="-ml-px inline-flex flex-none items-center justify-center gap-2 rounded-r-lg border border-teal-700 bg-teal-700 px-3 py-2 text-sm font-semibold leading-5 text-white hover:border-teal-600 hover:bg-teal-600 hover:text-white focus:ring focus:ring-teal-400/50 active:border-teal-700 active:bg-teal-700 dark:focus:ring-teal-400/90"
                        >
                          Subscribe
                        </button>
                      </div>
                    </form>
                  </div>
                </div>
                <hr className="my-10 border-dashed dark:border-gray-700/75" />
                <div className="flex flex-col gap-6 text-center text-sm md:flex-row-reverse md:justify-between md:gap-0 md:text-left">
                  <nav className="space-x-4">
                    <a
                      href="#"
                      className="text-gray-400 hover:text-gray-800 dark:hover:text-white"
                    >
                      <svg
                        className="bi bi-twitter-x inline-block size-5"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="currentColor"
                        viewBox="0 0 16 16"
                        aria-hidden="true"
                      >
                        <path d="M12.6.75h2.454l-5.36 6.142L16 15.25h-4.937l-3.867-5.07-4.425 5.07H.316l5.733-6.57L0 .75h5.063l3.495 4.633L12.601.75Zm-.86 13.028h1.36L4.323 2.145H2.865l8.875 11.633Z" />
                      </svg>
                    </a>
                    <a href="#" className="text-gray-400 hover:text-[#1877f2]">
                      <svg
                        className="icon-facebook inline-block size-5"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path d="M9 8H6v4h3v12h5V12h3.642L18 8h-4V6.333C14 5.378 14.192 5 15.115 5H18V0h-3.808C10.596 0 9 1.583 9 4.615V8z" />
                      </svg>
                    </a>
                    <a href="#" className="text-gray-400 hover:text-[#405de6]">
                      <svg
                        className="icon-instagram inline-block size-5"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000-2.881z" />
                      </svg>
                    </a>
                    <a
                      href="#"
                      className="text-gray-400 hover:text-[#333] dark:hover:text-gray-50"
                    >
                      <svg
                        className="icon-github inline-block size-5"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
                      </svg>
                    </a>
                    <a href="#" className="text-gray-400 hover:text-[#ea4c89]">
                      <svg
                        className="icon-dribbble inline-block size-5"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path d="M12 0C5.372 0 0 5.373 0 12s5.372 12 12 12 12-5.373 12-12S18.628 0 12 0zm9.885 11.441c-2.575-.422-4.943-.445-7.103-.073a42.153 42.153 0 00-.767-1.68c2.31-1 4.165-2.358 5.548-4.082a9.863 9.863 0 012.322 5.835zm-3.842-7.282c-1.205 1.554-2.868 2.783-4.986 3.68a46.287 46.287 0 00-3.488-5.438A9.894 9.894 0 0112 2.087c2.275 0 4.368.779 6.043 2.072zM7.527 3.166a44.59 44.59 0 013.537 5.381c-2.43.715-5.331 1.082-8.684 1.105a9.931 9.931 0 015.147-6.486zM2.087 12l.013-.256c3.849-.005 7.169-.448 9.95-1.322.233.475.456.952.67 1.432-3.38 1.057-6.165 3.222-8.337 6.48A9.865 9.865 0 012.087 12zm3.829 7.81c1.969-3.088 4.482-5.098 7.598-6.027a39.137 39.137 0 012.043 7.46c-3.349 1.291-6.953.666-9.641-1.433zm11.586.43a41.098 41.098 0 00-1.92-6.897c1.876-.265 3.94-.196 6.199.196a9.923 9.923 0 01-4.279 6.701z" />
                      </svg>
                    </a>
                  </nav>
                  <div className="text-gray-500 dark:text-gray-400/80">
                    <span className="font-medium">Dehaze</span> ©
                  </div>
                </div>
              </div>
            </footer>
            {/* END Footer */}
          </main>
          {/* END Page Content */}
        </div>
        {/* END Page Container */}
      </>
    );
  }
  