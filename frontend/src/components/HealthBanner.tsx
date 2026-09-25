import type { Health } from "../types";

export default function HealthBanner({ health, error }: { health: Health | null; error: string | null }) {
  if (error) {
    return <div className="banner error">Cannot reach the API: {error}</div>;
  }
  if (!health || health.status === "ok") return null;
  if (health.missing_config.length) {
    return (
      <div className="banner warn">
        Missing configuration: <code>{health.missing_config.join(", ")}</code>. Add it to your <code>.env</code>{" "}
        file and restart the backend. Documents cannot be indexed and questions cannot be answered until then.
      </div>
    );
  }
  const broken = health.components.filter((component) => !component.ok);
  return (
    <div className="banner warn">
      Degraded: {broken.map((component) => `${component.name} (${component.detail ?? "unavailable"})`).join(", ")}
    </div>
  );
}
