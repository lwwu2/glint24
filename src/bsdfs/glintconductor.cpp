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
#include <mitsuba/hw/basicshader.h>
#include "microfacet.h"
#include "glint.h"
#include "ior.h"


//#define GLINT_G

MTS_NAMESPACE_BEGIN


class GlintConductor : public BSDF {
public:
    GlintConductor(const Properties &props) : BSDF(props) {
        ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();
        
        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(1.0f)));

        std::string materialName = props.getString("material", "Al");

        Spectrum intEta, intK;
        if (boost::to_lower_copy(materialName) == "none") {
            intEta = Spectrum(0.0f);
            intK = Spectrum(1.0f);
        } else {
            intEta.fromContinuousSpectrum(InterpolatedSpectrum(
                fResolver->resolve("data/ior/" + materialName + ".eta.spd")));
            intK.fromContinuousSpectrum(InterpolatedSpectrum(
                fResolver->resolve("data/ior/" + materialName + ".k.spd")));
        }

        Float extEta = lookupIOR(props, "extEta", "air");

        m_eta = props.getSpectrum("eta", intEta) / extEta;
        m_k   = props.getSpectrum("k", intK) / extEta;
        
        m_sigma = props.getFloat("sigma",2);
        m_fix_sigma = props.getBoolean("fix_sigma", false);
        m_res = props.getInteger("res",1024);
        m_texture_scale = props.getFloat("texture_scale",1);
        
        m_glint.initialize(Thread::getThread()->getFileResolver()->resolve(props.getString("micro_normal","")).string(),
                m_res,
                props.getFloat("thresh"),
                props.getFloat("alpha",0.04),
                props.getString("filter","box"));
    }

    GlintConductor(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_alphaUV = static_cast<Texture2D *>(manager->getInstance(stream));
        m_rot = static_cast<Texture2D *>(manager->getInstance(stream));
        
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_eta = Spectrum(stream);
        m_k = Spectrum(stream);
        
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);
        
        manager->serialize(stream, m_alphaUV.get());
        
        manager->serialize(stream, m_specularReflectance.get());
        m_eta.serialize(stream);
        m_k.serialize(stream);
    }

    void configure() {
        unsigned int extraFlags = 0;
        extraFlags |= EAnisotropic;

        //if (!m_specularReflectance->isConstant())
        extraFlags |= ESpatiallyVarying;

        m_components.clear();
        m_components.push_back(EGlossyReflection | EFrontSide | extraFlags);

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);

        //m_usesRayDifferentials =
        //    m_alphaUV->usesRayDifferentials() ||
        //    m_specularReflectance->usesRayDifferentials();
        m_usesRayDifferentials = true;
        
        if (m_alphaUV && m_rot) {
            m_use_G = true;
        } else {
            m_use_G = false;
        }
        
        BSDF::configure();
    }

    /// Helper function: reflect \c wi with respect to a given surface normal
    inline Vector reflect(const Vector &wi, const Normal &m) const {
        return 2 * dot(wi, m) * Vector(m) - wi;
    }
    

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        /* Stop if this component was not requested */
        if (measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0 ||
            ((bRec.component != -1 && bRec.component != 0) ||
            !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.0f);

        /* Calculate the reflection half-vector */
        Vector H = normalize(bRec.wo+bRec.wi);
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
        
        /* Evaluate the microfacet normal distribution */
        Float D = m_glint.eval(bRec.its.uv*m_texture_scale,sigmaU,sigmaV,H);
        
        if (D == 0)
            return Spectrum(0.0f);

        /* Fresnel factor */
        const Spectrum F = fresnelConductorExact(dot(bRec.wi, H), m_eta, m_k) *
            m_specularReflectance->eval(bRec.its);

        /* Smith's shadow-masking function */
        Float G;
#ifdef GLINT_G
        G = m_glint.G(bRec.its.uv*m_texture_scale, sigmaU, sigmaV,bRec.wi, bRec.wo, H);
#else
        if (m_use_G) {
            duvdx *= (m_sigma/(float)m_res);
            duvdy *= (m_sigma/(float)m_res);
            Spectrum alphaUV;
            alphaUV = m_alphaUV->eval(bRec.its.uv*m_texture_scale,duvdx,duvdy);
            Float alphaU = std::max(std::min(alphaUV[0],1.0f),1e-4f);
            Float alphaV = std::max(std::min(alphaUV[1],1.0f),1e-4f);
            alphaUV =  m_rot->eval(bRec.its.uv*m_texture_scale,duvdx,duvdy);
            Vector2 r0(alphaUV[0],alphaUV[1]);
            r0 /= r0.length();
            Vector2 r1(-r0.y,r0.x);
            Vector wi(bRec.wi.x*r0.x+bRec.wi.y*r1.x,bRec.wi.x*r0.y+bRec.wi.y*r1.y,bRec.wi.z);
            Vector wo(bRec.wo.x*r0.x+bRec.wo.y*r1.x,bRec.wo.x*r0.y+bRec.wo.y*r1.y,bRec.wo.z);
            Vector wm = normalize(wo+wi);
            G = m_glint.G(wi,wo,wm, alphaU, alphaV);
        } else {
            G = m_glint.G(bRec.wi, bRec.wo, H);
        }
#endif        

        /* Calculate the total amount of reflection */
        Float model = D * G / (4.0f * Frame::cosTheta(bRec.wi));

        return F * model;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        if (measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0 ||
            ((bRec.component != -1 && bRec.component != 0) ||
            !(bRec.typeMask & EGlossyReflection)))
            return 0.0f;

        /* Calculate the reflection half-vector */
        Vector H = normalize(bRec.wo+bRec.wi);
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
        
        return m_glint.pdf(bRec.its.uv*m_texture_scale,sigmaU,sigmaV, H) / (4 * absDot(bRec.wo, H));
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        if (Frame::cosTheta(bRec.wi) < 0 ||
            ((bRec.component != -1 && bRec.component != 0) ||
            !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.0f);

        
        /* Sample M, the microfacet normal */
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
        Normal m = m_glint.sample(bRec.its.uv*m_texture_scale,sigmaU,sigmaV, sample);
        

        /* Perfect specular reflection based on the microfacet normal */
        bRec.wo = reflect(bRec.wi, m);
        bRec.eta = 1.0f;
        bRec.sampledComponent = 0;
        bRec.sampledType = EGlossyReflection;

        /* Side check */
        if (Frame::cosTheta(bRec.wo) <= 0)
            return Spectrum(0.0f);

        Spectrum F = fresnelConductorExact(dot(bRec.wi, m),
            m_eta, m_k) * m_specularReflectance->eval(bRec.its);
        
        Float G;
#ifdef GLINT_G
        G = m_glint.G(bRec.its.uv*m_texture_scale, sigmaU, sigmaV,bRec.wi, bRec.wo, m);
#else
        if (m_use_G) {
            duvdx *= (m_sigma/(float)m_res);
            duvdy *= (m_sigma/(float)m_res);
            Spectrum alphaUV;
            alphaUV = m_alphaUV->eval(bRec.its.uv*m_texture_scale,duvdx,duvdy);
            Float alphaU = std::max(std::min(alphaUV[0],1.0f),1e-4f);
            Float alphaV = std::max(std::min(alphaUV[1],1.0f),1e-4f);
            alphaUV =  m_rot->eval(bRec.its.uv*m_texture_scale,duvdx,duvdy);
            Vector2 r0(alphaUV[0],alphaUV[1]);
            r0 /= r0.length();
            Vector2 r1(-r0.y,r0.x);
            Vector wi(bRec.wi.x*r0.x+bRec.wi.y*r1.x,bRec.wi.x*r0.y+bRec.wi.y*r1.y,bRec.wi.z);
            Vector wo(bRec.wo.x*r0.x+bRec.wo.y*r1.x,bRec.wo.x*r0.y+bRec.wo.y*r1.y,bRec.wo.z);
            Vector wm = normalize(wo+wi);
            G = m_glint.G(wi,wo,wm, alphaU, alphaV);
        } else {
            G = m_glint.G(bRec.wi, bRec.wo, m);
        }
#endif
        Float weight;
        weight = G*dot(bRec.wi, m) 
               / (Frame::cosTheta(m) * Frame::cosTheta(bRec.wi));
        

        return F * weight;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
        if (Frame::cosTheta(bRec.wi) < 0 ||
            ((bRec.component != -1 && bRec.component != 0) ||
            !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.0f);
        
        /* Sample M, the microfacet normal */
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
        //printf("%f\n",sigmaU);

        /* Sample M, the microfacet normal */
        sigmaU = std::min(sigmaU,128.0f);
        sigmaV = std::min(sigmaV,128.0f);
        Normal m = m_glint.sample(bRec.its.uv*m_texture_scale,sigmaU,sigmaV, sample, pdf);

        if (pdf == 0)
            return Spectrum(0.0f);

        /* Perfect specular reflection based on the microfacet normal */
        bRec.wo = reflect(bRec.wi, m);
        bRec.eta = 1.0f;
        bRec.sampledComponent = 0;
        bRec.sampledType = EGlossyReflection;

        /* Side check */
        if (Frame::cosTheta(bRec.wo) <= 0)
            return Spectrum(0.0f);

        Spectrum F = fresnelConductorExact(dot(bRec.wi, m),
            m_eta, m_k) * m_specularReflectance->eval(bRec.its);

        Float weight;
        Float G;
#ifdef GLINT_G
        G = m_glint.G(bRec.its.uv*m_texture_scale, sigmaU, sigmaV,bRec.wi, bRec.wo, m);
#else
        if (m_use_G) {
            duvdx *= (m_sigma/(float)m_res);
            duvdy *= (m_sigma/(float)m_res);
            Spectrum alphaUV;
            alphaUV = m_alphaUV->eval(bRec.its.uv*m_texture_scale,duvdx,duvdy);
            Float alphaU = std::max(std::min(alphaUV[0],1.0f),1e-4f);
            Float alphaV = std::max(std::min(alphaUV[1],1.0f),1e-4f);
            alphaUV =  m_rot->eval(bRec.its.uv*m_texture_scale,duvdx,duvdy);
            Vector2 r0(alphaUV[0],alphaUV[1]);
            r0 /= r0.length();
            Vector2 r1(-r0.y,r0.x);
            Vector wi(bRec.wi.x*r0.x+bRec.wi.y*r1.x,bRec.wi.x*r0.y+bRec.wi.y*r1.y,bRec.wi.z);
            Vector wo(bRec.wo.x*r0.x+bRec.wo.y*r1.x,bRec.wo.x*r0.y+bRec.wo.y*r1.y,bRec.wo.z);
            Vector wm = normalize(wo+wi);
            G = m_glint.G(wi,wo,wm, alphaU, alphaV);
        } else {
            G = m_glint.G(bRec.wi, bRec.wo, m);
        }
#endif
        weight = G*dot(bRec.wi, m) 
               / (Frame::cosTheta(m) * Frame::cosTheta(bRec.wi));

        /* Jacobian of the half-direction mapping */
        pdf /= 4.0f * dot(bRec.wo, m);

        return F * weight;
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (name == "alphaUV")
                m_alphaUV = static_cast<Texture2D *>(child);
            else if (name == "rot")
                m_rot = static_cast<Texture2D *>(child);
            else if (name == "specularReflectance")
                m_specularReflectance = static_cast<Texture *>(child);
            else
                BSDF::addChild(name, child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    Float getRoughness(const Intersection &its, int component) const {
        Spectrum alphaUV = m_alphaUV->eval(its.uv*m_texture_scale);
        return 0.5f*(alphaUV[0]+alphaUV[1]);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "GlintConductor[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  distribution = glint" << "," << endl
            << "  alphaUV = " << indent(m_alphaUV->toString()) << "," << endl
            << "  sigma = " << m_sigma << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  eta = " << m_eta.toString() << "," << endl
            << "  k = " << m_k.toString() << endl
            << "]";
        return oss.str();
    }

    Shader *createShader(Renderer *renderer) const;

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_specularReflectance;
    ref<Texture2D> m_alphaUV;
    ref<Texture2D> m_rot;
    Spectrum m_eta, m_k;
    bool m_use_G;
    bool m_fix_sigma;
    Float m_sigma;
    Float m_res;
    Float m_texture_scale;
    GlintDistribution m_glint;
};






















/**
 * GLSL port of the rough conductor shader. This version is much more
 * approximate -- it only supports the Ashikhmin-Shirley distribution,
 * does everything in RGB, and it uses the Schlick approximation to the
 * Fresnel reflectance of conductors. When the roughness is lower than
 * \alpha < 0.2, the shader clamps it to 0.2 so that it will still perform
 * reasonably well in a VPL-based preview.
 */
class GlintConductorShader : public Shader {
public:
    GlintConductorShader(Renderer *renderer, const Texture *specularReflectance,
            const Texture *alphaU, const Texture *alphaV, const Spectrum &eta,
            const Spectrum &k) : Shader(renderer, EBSDFShader),
            m_specularReflectance(specularReflectance), m_alphaU(alphaU), m_alphaV(alphaV) {
        m_specularReflectanceShader = renderer->registerShaderForResource(m_specularReflectance.get());
        m_alphaUShader = renderer->registerShaderForResource(m_alphaU.get());
        m_alphaVShader = renderer->registerShaderForResource(m_alphaV.get());

        /* Compute the reflectance at perpendicular incidence */
        m_R0 = fresnelConductorExact(1.0f, eta, k);
    }

    bool isComplete() const {
        return m_specularReflectanceShader.get() != NULL &&
               m_alphaUShader.get() != NULL &&
               m_alphaVShader.get() != NULL;
    }

    void putDependencies(std::vector<Shader *> &deps) {
        deps.push_back(m_specularReflectanceShader.get());
        deps.push_back(m_alphaUShader.get());
        deps.push_back(m_alphaVShader.get());
    }

    void cleanup(Renderer *renderer) {
        renderer->unregisterShaderForResource(m_specularReflectance.get());
        renderer->unregisterShaderForResource(m_alphaU.get());
        renderer->unregisterShaderForResource(m_alphaV.get());
    }

    void resolve(const GPUProgram *program, const std::string &evalName, std::vector<int> &parameterIDs) const {
        parameterIDs.push_back(program->getParameterID(evalName + "_R0", false));
    }

    void bind(GPUProgram *program, const std::vector<int> &parameterIDs, int &textureUnitOffset) const {
        program->setParameter(parameterIDs[0], m_R0);
    }

    void generateCode(std::ostringstream &oss,
            const std::string &evalName,
            const std::vector<std::string> &depNames) const {
        oss << "uniform vec3 " << evalName << "_R0;" << endl
            << endl
            << "float " << evalName << "_D(vec3 m, float alphaU, float alphaV) {" << endl
            << "    float ct = cosTheta(m), ds = 1-ct*ct;" << endl
            << "    if (ds <= 0.0)" << endl
            << "        return 0.0f;" << endl
            << "    alphaU = 2 / (alphaU * alphaU) - 2;" << endl
            << "    alphaV = 2 / (alphaV * alphaV) - 2;" << endl
            << "    float exponent = (alphaU*m.x*m.x + alphaV*m.y*m.y)/ds;" << endl
            << "    return sqrt((alphaU+2) * (alphaV+2)) * 0.15915 * pow(ct, exponent);" << endl
            << "}" << endl
            << endl
            << "float " << evalName << "_G(vec3 m, vec3 wi, vec3 wo) {" << endl
            << "    if ((dot(wi, m) * cosTheta(wi)) <= 0 || " << endl
            << "        (dot(wo, m) * cosTheta(wo)) <= 0)" << endl
            << "        return 0.0;" << endl
            << "    float nDotM = cosTheta(m);" << endl
            << "    return min(1.0, min(" << endl
            << "        abs(2 * nDotM * cosTheta(wo) / dot(wo, m))," << endl
            << "        abs(2 * nDotM * cosTheta(wi) / dot(wi, m))));" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "_schlick(float ct) {" << endl
            << "    float ctSqr = ct*ct, ct5 = ctSqr*ctSqr*ct;" << endl
            << "    return " << evalName << "_R0 + (vec3(1.0) - " << evalName << "_R0) * ct5;" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "   if (cosTheta(wi) <= 0 || cosTheta(wo) <= 0)" << endl
            << "        return vec3(0.0);" << endl
            << "   vec3 H = normalize(wi + wo);" << endl
            << "   vec3 reflectance = " << depNames[0] << "(uv);" << endl
            << "   float alphaU = max(0.2, " << depNames[1] << "(uv).r);" << endl
            << "   float alphaV = max(0.2, " << depNames[2] << "(uv).r);" << endl
            << "   float D = " << evalName << "_D(H, alphaU, alphaV)" << ";" << endl
            << "   float G = " << evalName << "_G(H, wi, wo);" << endl
            << "   vec3 F = " << evalName << "_schlick(1-dot(wi, H));" << endl
            << "   return reflectance * F * (D * G / (4*cosTheta(wi)));" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
            << "        return vec3(0.0);" << endl
            << "    return " << evalName << "_R0 * inv_pi * inv_pi * cosTheta(wo);"<< endl
            << "}" << endl;
    }
    MTS_DECLARE_CLASS()
private:
    ref<const Texture> m_specularReflectance;
    ref<const Texture> m_alphaU;
    ref<const Texture> m_alphaV;
    ref<Shader> m_specularReflectanceShader;
    ref<Shader> m_alphaUShader;
    ref<Shader> m_alphaVShader;
    Spectrum m_R0;
};

Shader *GlintConductor::createShader(Renderer *renderer) const {
    return new GlintConductorShader(renderer,
        m_specularReflectance.get(), m_alphaUV.get(), m_alphaUV.get(), m_eta, m_k);
}

MTS_IMPLEMENT_CLASS(GlintConductorShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(GlintConductor, false, BSDF)
MTS_EXPORT_PLUGIN(GlintConductor, "Glint conductor BRDF");
MTS_NAMESPACE_END
