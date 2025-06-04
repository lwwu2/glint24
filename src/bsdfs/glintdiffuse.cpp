/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

MTS_NAMESPACE_BEGIN

inline int sign(float x) { 
    int t = x<0 ? -1 : 0;
    return x > 0 ? 1 : t;
}


void clip_arc(Point2 p0, Point2 p1, const float c, Point3* p) {
    // p: [y0,0: not clip, 1: clip, -1: drop],[y1,0: not clip, 1: clip, -1: drop]
    const float x0=p0.x;
    const float y0=p0.y;
    const float x1=p1.x;
    const float y1=p1.y;
    
    float Dx = x1-x0;
    float Dy = y1-y0;
    
    float C = c*c*(y0*y0-1)+x0*x0;
    
    if (C>0 && x0<0) {// p0 needs to be clipped
        float A = (c*c*Dy*Dy+Dx*Dx);
        A = std::max(A,1e-12f);
        
        float B = 2*(c*c*y0*Dy+x0*Dx);
        float D = B*B-4*A*C;
        float sqrtD = sqrtf(std::max(D,0.0f));
        float t0 = (-B-sqrtD)/(2*A);
        float t1 = (-B+sqrtD)/(2*A);
        
        if (c*c*(y1*y1-1)+x1*x1>0 && x1<0) {// both need to be clipped
            if (D<0||t0<0||t0>1||t1<0||t1>1) {// no intersection
                p[0].z = -1;
                p[1].z = -1;
                return;
            }
            
            p[0].x = Dx*t0+x0;
            p[0].y = Dy*t0+y0;
            p[0].z = 1;
            p[1].x = Dx*t1+x0;
            p[1].y = Dy*t1+y0;
            p[1].z = 1;
            return;
        } 
        // only clip p0
        t0 = std::max(std::min(t0,1.0f),0.0f);
        p[0].x = Dx*t0+x0;
        p[0].y = Dy*t0+y0;
        p[0].z = 1;
        p[1].x = x1;
        p[1].y = y1;
        p[1].z = 0;
        return;
    } else {
        float A = (c*c*Dy*Dy+Dx*Dx);
        A = std::max(A,1e-12f);
    
        float B = 2*(c*c*y0*Dy+x0*Dx);
        float D = B*B-4*A*C;
        float sqrtD = sqrtf(std::max(D,0.0f));
        //float t0 = (-B-sqrtD)/(2*A);
        float t1 = (-B+sqrtD)/(2*A);
        
        if (c*c*(y1*y1-1)+x1*x1>0 && x1<0) {// clip p1
            t1 = std::max(std::min(t1,1.0f),0.0f);
            p[1].x = Dx*t1+x0;
            p[1].y = Dy*t1+y0;
            p[1].z = 1;
            p[0].x = x0;
            p[0].y = y0;
            p[0].z = 0;
            return;
        }
    }
    p[0].x = x0;
    p[0].y = y0;
    p[0].z = 0;
    p[1].x = x1;
    p[1].y = y1;
    p[1].z = 0;
    return;
}

inline float int_arc(float y0, float y1) {
    return -0.5*((std::asin(y1)-std::asin(y0))+(y1*std::sqrt(1-y1*y1)-y0*std::sqrt(1-y0*y0)));
}

inline float int_line(float x0, float y0, float x1, float y1, const float c) {
    float Dy = y1-y0;
    float Dx = x1-x0;
    float r = std::sqrt(Dx*Dx+Dy*Dy);
    if (r<1e-6) { //point zero
        return 0.0f;
    }
    Dx/=r;Dy/=r;
    r = Dy*x0-Dx*y0;
    r = std::sqrt(std::max(1-r*r,0.0f));
    
    float p0 = (Dx*x0+Dy*y0)/r;
    float p1 = (Dx*x1+Dy*y1)/r;
    
    float disk = Dy*r*r*(p1*std::sqrt(1-p1*p1)-p0*std::sqrt(1-p0*p0)+std::asin(p1)-std::asin(p0));
    disk = std::abs(disk)*sign(y1-y0);
    float cube = (x1+x0)*(y1-y0);
    return 0.5*(c*cube-std::sqrt(1-c*c)*disk);
}



Float project_triangle(
    Point2 &p0, Point2 &p1, Point2 &p2,
    const float c
) {
    // clip all the edges
    Point3 p[6];// 3 edges in total
    uint32_t front = 0;
    clip_arc(p0, p1, c, p);
    if (p[front*2].z>=0) {// not drop current edge
        front += 1;
    }
    clip_arc(p1, p2, c, p+front*2);
    if (p[front*2].z>=0) {// not drop current edge
        front += 1;
    }
    clip_arc(p2, p0, c, p+front*2);
    if (p[front*2].z>=0) {// not drop current edge
        front += 1;
    }
    
    // eval all integrals
    Float result = 0.0f;
    for (uint32_t i=0; i < front; i++) {
        result += int_line(p[i*2].x, p[i*2].y, p[i*2+1].x, p[i*2+1].y,c); // line integral
        
        if (p[i*2+1].z>0 && p[((i+1)%front)*2].z>0) { // need to integate arc
            result += int_arc(p[i*2+1].y,p[((i+1)%front)*2].y);
        }
    }
    return std::abs(result);
}



class GlintDiffuse : public BSDF {
public:
    GlintDiffuse(const Properties &props)
        : BSDF(props) {
        /* For better compatibility with other models, support both
           'reflectance' and 'diffuseReflectance' as parameter names */
        m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
            props.hasProperty("reflectance") ? "reflectance"
                : "diffuseReflectance", Spectrum(.5f)));
                
                
        m_sigma = props.getFloat("sigma",2);
        m_fix_sigma = props.getBoolean("fix_sigma", false);
        m_res = props.getInteger("res",1024);
        m_texture_scale = props.getFloat("texture_scale",1);
        
        std::string fileName = Thread::getThread()->getFileResolver()->resolve(props.getString("micro_normal","")).string();
        
        m_micro_normal = new float[m_res*m_res*2];
        if (!m_micro_normal) printf("\nCannot allocate memory!\n");
        
        std::FILE *fp = std::fopen((fileName+"/normal").c_str(), "rb");
        std::fread(m_micro_normal,sizeof(float),m_res*m_res*2,fp);
        std::fclose(fp);
        printf("OK\n");
        
    }

    GlintDiffuse(Stream *stream, InstanceManager *manager)
        : BSDF(stream, manager) {
        m_reflectance = static_cast<Texture *>(manager->getInstance(stream));

        configure();
    }

    void configure() {
        /* Verify the input parameter and fix them if necessary */
        m_reflectance = ensureEnergyConservation(m_reflectance, "reflectance", 1.0f);

        m_components.clear();
        //if (m_reflectance->getMaximum().max() > 0)
        //    m_components.push_back(EDiffuseReflection | EFrontSide
        //        | ESpatiallyVarying | EAnisotropic);
        //    m_usesRayDifferentials = m_reflectance->usesRayDifferentials();
        m_components.push_back(EDiffuseReflection | EFrontSide | ESpatiallyVarying | EAnisotropic);
        m_usesRayDifferentials = true;
        
        BSDF::configure();
    }

    Spectrum getDiffuseReflectance(const Intersection &its) const {
        return m_reflectance->eval(its);
    }
    
    
    inline Point2 fetch(int32_t i, int32_t j) const {
        i = math::modulo(i,m_res);
        j = math::modulo(j,m_res);
        //i %= m_res;
        //j %= m_res;
        return Point2(m_micro_normal[i*m_res*2+j*2],
                      m_micro_normal[i*m_res*2+j*2+1]);
    }
    
    inline void fetch4(int32_t i0, int32_t j0, Point2* n) const {
        int32_t i1 = i0+1;
        int32_t j1 = j0+1;
        n[0] = fetch(i0,j0);
        n[1] = fetch(i0,j1);
        n[2] = fetch(i1,j0);
        n[3] = fetch(i1,j1);
        return;
    }
    
    inline Float area3(Point2 &p1, Point2 &p2, Point2 &p3) const {
        return (p1.x-p3.x)*(p2.y-p3.y)
             - (p1.y-p3.y)*(p2.x-p3.x);
    }
    
    
    Float Gsmith(const Point2 &uv_in, const Float sigmaU, const Float sigmaV,
        const Vector wi, const Vector wo 
    ) const {
        if (Frame::cosTheta(wi)<=0||Frame::cosTheta(wo)<=0) {
            return 0.0f;
        }
        Point2 uv(fmod(uv_in.x,1.0f)*m_res,fmod(uv_in.y,1.0f)*m_res);
        
        // base mip level
        int32_t u0 = (int32_t)floor((uv.x-sigmaU));
        int32_t v0 = (int32_t)floor((uv.y-sigmaV));
        int32_t u1 = ((int32_t)ceil((uv.x+sigmaU)))-1;
        int32_t v1 = ((int32_t)ceil((uv.y+sigmaV)))-1;
        
        float phi_i = 0.0f;
        float phi_o = 0.0f;
        if (wi.z < 0.99999f) {
            phi_i = std::atan2(wi.y,wi.x);
            phi_o = std::atan2(wo.y,wo.x);
        }
        Vector2 To,Ti;
        math::sincos(phi_i, &Ti.y, &Ti.x);
        math::sincos(phi_o, &To.y, &To.x);
        
        float Pi=0.0f;
        float Po=0.0f;
        Point2 n[4];
        Point2 m[4];
        for (int32_t dv=v0;dv<=v1;dv++) {
            for (int32_t du=u0;du<=u1;du++) {
                fetch4(dv,du,n);
                Float area0 = abs(area3(n[0],n[1],n[2]));
                Float area1 = abs(area3(n[2],n[1],n[3]));
                Float nz = std::sqrt(std::max(1-n[1].x*n[1].x-n[1].y*n[1].y,0.0f));
                
                // project wi
                for (int i=0; i<4; i++) {
                    m[i].x = Ti.x*n[i].x+Ti.y*n[i].y;
                    m[i].y = Ti.x*n[i].y-Ti.y*n[i].x;
                }
                if (area0>1e-6) {
                    Pi += project_triangle(m[0],m[1],m[2],wi.z)/area0;
                } else {
                    Pi += 0.5f*std::max(wi.x*n[1].x+wi.y*n[1].y+wi.z*nz,0.0f);
                }
                if (area1>1e-6) {
                    Pi += project_triangle(m[2],m[1],m[3],wi.z)/area1;
                } else {
                    Pi += 0.5f*std::max(wi.x*n[1].x+wi.y*n[1].y+wi.z*nz,0.0f);
                }
                
                // project wo
                for (int i=0; i<4; i++) {
                    m[i].x = To.x*n[i].x+To.y*n[i].y;
                    m[i].y = To.x*n[i].y-To.y*n[i].x;
                }
                if (area0>1e-6) {
                    Po += project_triangle(m[0],m[1],m[2],wo.z)/area0;
                } else {
                    Po += 0.5f*std::max(wo.x*n[1].x+wo.y*n[1].y+wo.z*nz,0.0f);
                }
                if (area1>1e-6) {
                    Po += project_triangle(m[2],m[1],m[3],wo.z)/area1;
                } else {
                    Po += 0.5f*std::max(wo.x*n[1].x+wo.y*n[1].y+wo.z*nz,0.0f);
                }
            }
        }
        
        Pi /= (v1-v0+1)*(u1-u0+1);
        Po /= (v1-v0+1)*(u1-u0+1);
        
        Pi = std::min(std::max(Pi,Frame::cosTheta(wi)),1.0f);
        Po = std::min(std::max(Po,0.0f),4.0f);
        
        return Po/Frame::cosTheta(wo);
        
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) <= 0
            || Frame::cosTheta(bRec.wo) <= 0)
            return Spectrum(0.0f);

        Vector2f duvdx(bRec.its.dudx,bRec.its.dvdx);
        Vector2f duvdy(bRec.its.dudy,bRec.its.dvdy);
        if (m_fix_sigma) {
            duvdx.x = 2.0f;duvdx.y =0.0f;
            duvdy.x = 0.0f;duvdy.y =2.0f;
        }
        float sigmaU = (fabsf(duvdx.x)+fabsf(duvdy.x))/2.0f;
        float sigmaV = (fabsf(duvdx.y)+fabsf(duvdy.y))/2.0f;
        sigmaU *= m_sigma;
        sigmaV *= m_sigma;
        sigmaU = std::max(1.0f,sigmaU);
        sigmaV = std::max(1.0f,sigmaV);
        sigmaU = std::min(sigmaU,128.0f);
        sigmaV = std::min(sigmaV,128.0f);
        Point2 uv_in = bRec.its.uv*m_texture_scale;
        Float G = Gsmith(uv_in,sigmaU,sigmaV,bRec.wi, bRec.wo);
        
        
        return m_reflectance->eval(bRec.its)*G
            * (INV_PI * Frame::cosTheta(bRec.wo));
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) <= 0
            || Frame::cosTheta(bRec.wo) <= 0)
            return 0.0f;

        return warp::squareToCosineHemispherePdf(bRec.wo);
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(0.0f);

        bRec.wo = warp::squareToCosineHemisphere(sample);
        bRec.eta = 1.0f;
        bRec.sampledComponent = 0;
        bRec.sampledType = EDiffuseReflection;
        
        Vector2f duvdx(bRec.its.dudx,bRec.its.dvdx);
        Vector2f duvdy(bRec.its.dudy,bRec.its.dvdy);
        if (m_fix_sigma) {
            duvdx.x = 2.0f;duvdx.y =0.0f;
            duvdy.x = 0.0f;duvdy.y =2.0f;
        }
        float sigmaU = (fabsf(duvdx.x)+fabsf(duvdy.x))/2.0f;
        float sigmaV = (fabsf(duvdx.y)+fabsf(duvdy.y))/2.0f;
        sigmaU *= m_sigma;
        sigmaV *= m_sigma;
        sigmaU = std::max(1.0f,sigmaU);
        sigmaV = std::max(1.0f,sigmaV);
        sigmaU = std::min(sigmaU,128.0f);
        sigmaV = std::min(sigmaV,128.0f);
        Point2 uv_in = bRec.its.uv*m_texture_scale;
        Float G = Gsmith(uv_in,sigmaU,sigmaV,bRec.wi, bRec.wo);
        
        return m_reflectance->eval(bRec.its)*G;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
        if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(0.0f);

        bRec.wo = warp::squareToCosineHemisphere(sample);
        bRec.eta = 1.0f;
        bRec.sampledComponent = 0;
        bRec.sampledType = EDiffuseReflection;
        pdf = warp::squareToCosineHemispherePdf(bRec.wo);
        
        
        Vector2f duvdx(bRec.its.dudx,bRec.its.dvdx);
        Vector2f duvdy(bRec.its.dudy,bRec.its.dvdy);
        if (m_fix_sigma) {
            duvdx.x = 2.0f;duvdx.y =0.0f;
            duvdy.x = 0.0f;duvdy.y =2.0f;
        }
        float sigmaU = (fabsf(duvdx.x)+fabsf(duvdy.x))/2.0f;
        float sigmaV = (fabsf(duvdx.y)+fabsf(duvdy.y))/2.0f;
        sigmaU *= m_sigma;
        sigmaV *= m_sigma;
        sigmaU = std::max(1.0f,sigmaU);
        sigmaV = std::max(1.0f,sigmaV);
        sigmaU = std::min(sigmaU,128.0f);
        sigmaV = std::min(sigmaV,128.0f);
        Point2 uv_in = bRec.its.uv*m_texture_scale;
        Float G = Gsmith(uv_in,sigmaU,sigmaV,bRec.wi, bRec.wo);
        
        return m_reflectance->eval(bRec.its)*G;
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))
                && (name == "reflectance" || name == "diffuseReflectance")) {
            m_reflectance = static_cast<Texture *>(child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_reflectance.get());
    }

    Float getRoughness(const Intersection &its, int component) const {
        return std::numeric_limits<Float>::infinity();
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "GlintDiffuse[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  reflectance = " << indent(m_reflectance->toString()) << endl
            << "]";
        return oss.str();
    }

    Shader *createShader(Renderer *renderer) const;

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_reflectance;
    Float m_texture_scale;
    Float m_sigma;
    uint32_t m_res;
    bool m_fix_sigma;
    float* m_micro_normal;
};

// ================ Hardware shader implementation ================

class GlintDiffuseShader : public Shader {
public:
    GlintDiffuseShader(Renderer *renderer, const Texture *reflectance)
        : Shader(renderer, EBSDFShader), m_reflectance(reflectance) {
        m_reflectanceShader = renderer->registerShaderForResource(m_reflectance.get());
    }

    bool isComplete() const {
        return m_reflectanceShader.get() != NULL;
    }

    void cleanup(Renderer *renderer) {
        renderer->unregisterShaderForResource(m_reflectance.get());
    }

    void putDependencies(std::vector<Shader *> &deps) {
        deps.push_back(m_reflectanceShader.get());
    }

    void generateCode(std::ostringstream &oss,
            const std::string &evalName,
            const std::vector<std::string> &depNames) const {
        oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
            << "        return vec3(0.0);" << endl
            << "    return " << depNames[0] << "(uv) * inv_pi * cosTheta(wo);" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "    return " << evalName << "(uv, wi, wo);" << endl
            << "}" << endl;
    }

    MTS_DECLARE_CLASS()
private:
    ref<const Texture> m_reflectance;
    ref<Shader> m_reflectanceShader;
};

Shader *GlintDiffuse::createShader(Renderer *renderer) const {
    return new GlintDiffuseShader(renderer, m_reflectance.get());
}

MTS_IMPLEMENT_CLASS(GlintDiffuseShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(GlintDiffuse, false, BSDF)
MTS_EXPORT_PLUGIN(GlintDiffuse, "Glint diffuse BRDF")
MTS_NAMESPACE_END
