import { NavLink } from "react-router-dom";

const navItems = [
  { label: "Upload", to: "/" },
  { label: "Analysis", to: "/analysis" },
  { label: "History", to: "/history" },
];

function NavBar() {
  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        <div className="text-lg font-semibold tracking-wide text-slate-900">VERIDEX</div>
        <nav className="flex items-center gap-2 rounded-lg bg-slate-100 p-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `rounded-md px-4 py-2 text-sm font-medium transition ${
                  isActive
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-600 hover:bg-white hover:text-slate-900"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}

export default NavBar;
