import { mat4, vec3 } from 'gl-matrix';

export class Camera {
  public position: vec3;
  public target: vec3;
  public up: vec3;
  
  public fov: number = Math.PI / 4;
  public aspect: number = 1;
  public near: number = 0.1;
  public far: number = 1000;

  private dragging: boolean = false;
  private lastMouse: { x: number, y: number } = { x: 0, y: 0 };
  
  private radius: number = 5;
  private theta: number = Math.PI / 4;
  private phi: number = Math.PI / 4;

  constructor(canvas: HTMLCanvasElement) {
    this.position = vec3.create();
    this.target = vec3.create();
    this.up = vec3.fromValues(0, 1, 0);

    this.updatePosition();

    canvas.addEventListener('mousedown', (e) => {
      this.dragging = true;
      this.lastMouse = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('mousemove', (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastMouse.x;
      const dy = e.clientY - this.lastMouse.y;
      
      this.theta -= dx * 0.01;
      this.phi += dy * 0.01;
      
      // Clamp phi to avoid flipping
      const eps = 0.01;
      this.phi = Math.max(eps, Math.min(Math.PI - eps, this.phi));
      
      this.updatePosition();
      this.lastMouse = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('mouseup', () => {
      this.dragging = false;
    });

    canvas.addEventListener('wheel', (e) => {
      this.radius += e.deltaY * 0.01;
      this.radius = Math.max(0.1, this.radius);
      this.updatePosition();
    });
  }

  private updatePosition() {
    this.position[0] = this.target[0] + this.radius * Math.sin(this.phi) * Math.sin(this.theta);
    this.position[1] = this.target[1] + this.radius * Math.cos(this.phi);
    this.position[2] = this.target[2] + this.radius * Math.sin(this.phi) * Math.cos(this.theta);
  }

  public getViewMatrix(): mat4 {
    const view = mat4.create();
    mat4.lookAt(view, this.position, this.target, this.up);
    return view;
  }

  public getProjectionMatrix(): mat4 {
    const proj = mat4.create();
    mat4.perspective(proj, this.fov, this.aspect, this.near, this.far);
    return proj;
  }
}
